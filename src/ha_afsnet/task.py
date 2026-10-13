"""Causal graph preparation, sampled loss, scores and gate values."""

import csv
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

from .evaluation import sample_negatives
from .model import HAAFSNet


class ForecastingTask:
    def __init__(self, data, config):
        self.data, self.config = data, config
        self.model = HAAFSNet(data.n, data.r, config).to(data.device)
        self.cache = OrderedDict()

    def graph(self, t, k):
        key = (t, k)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        rows = self.data.rows[t]
        edges = np.unique(rows[rows[:, 4] == k, :3], axis=0)
        src, rel, dst = edges.T
        packed = np.empty((0, 0), dtype=np.int64)
        active, indices, normalization = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
        if len(edges):
            pairs, reverse = np.unique(edges[:, [1, 0]], axis=0, return_inverse=True)
            active, starts, counts = np.unique(
                pairs[:, 0], return_index=True, return_counts=True
            )
            width = counts.max()
            packed = np.zeros((len(active), width), dtype=np.int64)
            positions = np.zeros(len(pairs), dtype=np.int64)
            for i, (start, count) in enumerate(zip(starts, counts)):
                packed[i, :count] = pairs[start : start + count, 1]
                positions[start : start + count] = i * width + np.arange(count)
            indices = positions[reverse]
            _, groups, counts = np.unique(
                edges[:, [2, 1]], axis=0, return_inverse=True, return_counts=True
            )
            normalization = (1 / counts[groups]).astype(np.float32)
        graph = {
            "source": src,
            "destination": dst,
            "relation": rel,
            "packed_source": packed,
            "active_relations": active,
            "message_index": indices,
            "normalization": normalization,
            "relation_count": np.bincount(rel, minlength=self.data.r)
            .clip(min=1)
            .astype(np.float32),
        }
        graph = {
            name: torch.tensor(value, device=self.data.device)
            for name, value in graph.items()
        }
        self.cache[key] = graph
        while len(self.cache) > 48:
            self.cache.popitem(last=False)
        return graph

    def situation_inputs(self, forecasts):
        w = self.config["history"]
        increments = []
        for t in forecasts:
            if t < w + 1:
                raise ValueError("Insufficient prefix for activity increments")
            increments.append(np.diff(self.data.activity[t - w - 1 : t]))
        return (
            torch.tensor(
                np.asarray(increments), dtype=torch.float32, device=self.data.device
            ),
            torch.tensor(
                self.data.rho[forecasts], dtype=torch.float32, device=self.data.device
            ),
        )

    def encode(self, t):
        dates = self.data.history(t, self.config["history"])
        if t < 2 * self.config["history"]:
            raise ValueError("Earliest fused snapshot needs prefix through t-2w")
        graphs = [[self.graph(date, k) for k in range(5)] for date in dates]
        increments, divergences = self.situation_inputs([date + 1 for date in dates])
        return self.model.encode(graphs, increments, divergences)

    def loss(self, t, rows, diagnostics=False):
        entity, relation, coefficients = self.encode(t)
        # Sample on CPU before transfer, retaining the same generator and draw order.
        cpu_triples = torch.tensor(rows[:, :3].copy())
        cpu_contexts = torch.tensor(rows[:, 4].copy())
        negatives = sample_negatives(
            cpu_triples,
            cpu_contexts,
            self.data.n,
            self.data.answers(t, contextual=True),
            torch.default_generator,
        ).to(self.data.device)
        triples = cpu_triples.to(self.data.device)
        contexts = cpu_contexts.to(self.data.device)
        # Decode each query once so positive and negative use the same dropout realization.
        query = self.model.decoder(
            entity[contexts, triples[:, 0]], relation[contexts, triples[:, 1]]
        )
        positive = (query * entity[contexts, triples[:, 2]]).sum(-1)
        negative = (query * entity[contexts, negatives]).sum(-1)
        prediction = F.softplus(-positive).sum() + F.softplus(negative).sum()
        regularization = (
            sum(parameter.square().sum() for parameter in self.model.parameters())
            * self.config["l2"]
        )
        loss = prediction + regularization
        if not diagnostics:
            return loss
        pos, neg, alpha = (
            positive.detach(),
            negative.detach(),
            coefficients[-1].detach(),
        )
        return loss, {
            "positive_bce_per_query": F.softplus(-pos).mean(),
            "negative_bce_per_query": F.softplus(neg).mean(),
            "l2_per_batch": regularization.detach(),
            "positive_logit_mean": pos.mean(),
            "negative_logit_mean": neg.mean(),
            "positive_above_sampled_negative_fraction": (pos > neg).float().mean(),
            "alpha_mean": alpha.mean(),
            "alpha_std": alpha.std(unbiased=False),
            "alpha_below_001_fraction": (alpha < 0.01).float().mean(),
            "alpha_above_099_fraction": (alpha > 0.99).float().mean(),
        }

    def training_batches(self, t, rows, diagnostics_interval=0, start_step=0):
        order = np.random.permutation(len(rows))
        for index, begin in enumerate(range(0, len(rows), self.config["batch_size"])):
            collect = bool(
                diagnostics_interval
                and (start_step + index + 1) % diagnostics_interval == 0
            )
            value = self.loss(
                t,
                rows[order[begin : begin + self.config["batch_size"]]],
                diagnostics=collect,
            )
            if diagnostics_interval:
                yield value if collect else (value, None)
            else:
                yield value

    def scores(self, t, rows):
        entity, relation, _ = self.encode(t)
        values = []
        for begin in range(0, len(rows), self.config["batch_size"]):
            batch = rows[begin : begin + self.config["batch_size"]]
            triples = torch.tensor(batch[:, :3].copy(), device=self.data.device)
            contexts = torch.tensor(batch[:, 4].copy(), device=self.data.device)
            values.append(self.model.scores(entity, relation, triples, contexts))
        return torch.cat(values)

    def export_gates(self, path, seed):
        self.model.eval()
        dates = [t for t in sorted(self.data.rows) if t >= self.config["history"] + 1]
        split_by_date = {
            t: split for split, times in self.data.splits.items() for t in times
        }
        fields = [
            "seed",
            "dataset",
            "split",
            "timestamp",
            "date",
            "context",
            "alpha",
            "activity",
            "activity_change",
            "composition_change",
            "diagnostic_row_activity",
        ]
        with open(path, "w") as stream, torch.no_grad():
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for begin in range(0, len(dates), 256):
                batch = dates[begin : begin + 256]
                increments, divergences = self.situation_inputs(batch)
                alpha = self.model.alpha(increments, divergences).cpu().numpy()
                for t, coefficients in zip(batch, alpha):
                    for k, coefficient in enumerate(coefficients):
                        writer.writerow(
                            {
                                "seed": seed,
                                "dataset": self.data.region,
                                "split": split_by_date[t],
                                "timestamp": t,
                                "date": self.data.dates[str(t)],
                                "context": k,
                                "alpha": float(coefficient),
                                "activity": self.data.activity[t - 1],
                                "activity_change": self.data.activity[t - 1]
                                - self.data.activity[t - 2],
                                "composition_change": self.data.rho[t],
                                "diagnostic_row_activity": self.data.row_activity[
                                    t - 1
                                ],
                            }
                        )
