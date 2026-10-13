"""Detached training signals without changing the objective or random sequence."""

import torch


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        root = name.split(".", 1)[0]
        group = (
            "graph"
            if root == "graph_layers"
            else "decoder"
            if root == "decoder"
            else "temporal"
            if root in {"transformer", "gru", "position"}
            else "attention"
            if root in {"entity_attention", "relation_attention"}
            else "embeddings"
            if root in {"entity", "relation", "context"}
            else "fusion"
        )
        norm = torch.linalg.vector_norm(parameter.grad.detach())
        groups.setdefault(group, []).append(norm.square())
    return {
        f"{name}_gradient_norm": torch.stack(values).sum().sqrt()
        for name, values in groups.items()
    }


class TrainingDiagnostics:
    """All-step global gradients and sampled batch/component signals."""

    def __init__(self, device, clip, state=None):
        self.device, self.clip = device, clip
        self.steps, self.samples = 0, 0
        self.norm_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.norm_max = self.norm_sum.clone()
        self.above_clip = self.norm_sum.clone()
        self.sample_sums = {}
        if state is not None:
            self.steps, self.samples = state["steps"], state["samples"]
            for key in ["norm_sum", "norm_max", "above_clip"]:
                setattr(
                    self,
                    key,
                    torch.tensor(state[key], dtype=torch.float64, device=device),
                )
            self.sample_sums = {
                key: torch.tensor(value, dtype=torch.float64, device=device)
                for key, value in state["sample_sums"].items()
            }

    def observe(self, norm, details=None):
        norm = norm.detach().double()
        self.steps += 1
        self.norm_sum += norm
        self.norm_max = torch.maximum(self.norm_max, norm)
        self.above_clip += (norm > self.clip).double()
        if details is not None:
            self.samples += 1
            for key, value in details.items():
                value = value.detach().double()
                if key not in self.sample_sums:
                    self.sample_sums[key] = torch.zeros_like(value)
                self.sample_sums[key] += value

    def state_dict(self):
        return {
            "steps": self.steps,
            "samples": self.samples,
            "norm_sum": float(self.norm_sum.cpu()),
            "norm_max": float(self.norm_max.cpu()),
            "above_clip": float(self.above_clip.cpu()),
            "sample_sums": {
                key: float(value.cpu()) for key, value in self.sample_sums.items()
            },
        }

    def summary(self):
        state = self.state_dict()
        return {
            "optimizer_steps": self.steps,
            "sampled_batches": self.samples,
            "pre_clip_gradient_norm_mean": state["norm_sum"] / max(self.steps, 1),
            "pre_clip_gradient_norm_max": state["norm_max"],
            "gradient_norm_above_clip_fraction": state["above_clip"]
            / max(self.steps, 1),
            "sampled_batch_means": {
                key: value / self.samples for key, value in state["sample_sums"].items()
            },
        }
