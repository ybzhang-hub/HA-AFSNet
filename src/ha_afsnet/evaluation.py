"""Shared full-vocabulary, cross-context, time-filtered tail ranking."""

import numpy as np
import torch


def filtered_ranks(scores, triples, answers, tie="average"):
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite candidate scores")
    values = scores.clone()
    for i, (s, r, o) in enumerate(triples.detach().cpu().tolist()):
        other = sorted(answers.get((s, r), set()) - {o})
        if other:
            values[i, other] = -torch.inf
    target = values.gather(1, triples[:, 2:3])
    greater = (values > target).sum(1)
    equal = (values == target).sum(1)
    if tie == "average":
        return 1 + greater.float() + 0.5 * (equal.float() - 1)
    if tie == "optimistic":
        return 1 + greater.float()
    if tie == "pessimistic":
        return greater.float() + equal.float()
    if tie == "entity_id":
        ids = torch.arange(values.shape[1], device=values.device)
        return (
            1
            + greater.float()
            + ((values == target) & (ids < triples[:, 2:3])).sum(1).float()
        )
    raise ValueError(tie)


def metrics(ranks):
    array = np.asarray(ranks, dtype=np.float64)
    if not array.size:
        raise ValueError("No retained evaluation queries")
    return {
        "MRR": float(np.mean(1 / array)),
        **{f"Hits@{k}": float(np.mean(array <= k)) for k in [1, 3, 10]},
        "retained_queries": int(array.size),
    }


def sample_negatives(triples, contexts, n, answers, generator):
    # Uniform sampling from the complement. No valid/test facts are consulted.
    negatives = []
    for (s, r, _), k in zip(triples.cpu().tolist(), contexts.cpu().tolist()):
        excluded = answers[(s, r, k)]
        if len(excluded) == n:
            raise ValueError("Query has no valid corrupted tail")
        value = int(torch.randint(n, (1,), generator=generator).item())
        while value in excluded:
            value = int(torch.randint(n, (1,), generator=generator).item())
        negatives.append(value)
    return torch.tensor(negatives, device=triples.device)


def evaluate(
    adapter,
    data,
    split,
    tie,
    limit_dates=0,
    limit_queries=0,
    save=None,
    progress=None,
    collector=None,
):
    adapter.model.eval()
    dates = data.splits[split][: limit_dates or None]
    ranks, identifiers = [], []
    with torch.no_grad():
        for i, t in enumerate(dates):
            rows = data.rows[t][: limit_queries or None]
            # Full target vocabulary is preserved. Popularity removes queries only.
            scores = adapter.scores(t, rows)
            triples = torch.tensor(rows[:, :3].copy(), device=data.device)
            current = (
                filtered_ranks(scores, triples, data.answers(t), tie).cpu().numpy()
            )
            mask = np.array([int(o) not in data.heads for o in rows[:, 2]])
            ranks.extend(current[mask].tolist())
            identifiers.extend(rows[mask].tolist())
            if progress is not None and ((i + 1) % 25 == 0 or i + 1 == len(dates)):
                progress(
                    {
                        "split": split,
                        "completed_dates": i + 1,
                        "total_dates": len(dates),
                        "timestamp": t,
                        "retained_queries_so_far": len(ranks),
                    }
                )
    if save:
        np.savez_compressed(
            save, ranks=np.asarray(ranks), queries=np.asarray(identifiers)
        )
    if collector is not None:
        collector.update(ranks=np.asarray(ranks), queries=np.asarray(identifiers))
    return metrics(ranks)
