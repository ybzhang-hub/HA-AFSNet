"""Project locations and supported scientific configurations."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(os.environ.get("HA_AFSNET_DATA_ROOT", PROJECT_ROOT / "data"))
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "cache/prepared"

PROTOCOL_FIELDS = (
    "query_direction",
    "filter_key",
    "filter_context",
    "tie_handling",
    "negative_exclusions",
    "candidate_entities",
    "popularity",
    "history",
    "context",
    "checkpoint",
    "seeds",
)


def canonical_protocol(protocol):
    """Select the fields that define the evaluation protocol."""
    return {key: protocol[key] for key in PROTOCOL_FIELDS}


def canonical_config(config):
    config = dict(config)
    config.pop("implementation_status", None)
    if (
        config.get("objective")
        == "equation_37_positive_plus_mean_negative_BCE_and_explicit_l2"
    ):
        config["objective"] = "sampled_bce_sum_with_l2"
    return config


def validate_protocol(protocol):
    supported = {
        "query_direction": "tail_only",
        "filter_key": ["subject", "relation", "timestamp"],
        "filter_context": False,
        "negative_exclusions": "training_true_tails_same_subject_relation_context_date",
        "candidate_entities": "full_released_vocabulary",
        "popularity": "released_head_ents_train_occurrence_cumulative_one_third",
        "history": "rolling_observed_prefix_strictly_before_query_date",
        "context": "released_LDA_K5_event_mean_article_posterior_argmax",
        "checkpoint": "validation_MRR; >= best updates checkpoint and resets patience",
        "seeds": [42, 43, 44, 45, 46],
    }
    for key, expected in supported.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Unsupported protocol {key}: expected {expected!r}")
    if protocol.get("tie_handling") not in [
        "average",
        "optimistic",
        "pessimistic",
        "entity_id",
    ]:
        raise ValueError("Unsupported tie handling")


def validate_config(model, config):
    allowed = {
        "model",
        "hidden",
        "attention_width",
        "context_width",
        "activity_width",
        "composition_width",
        "situation_width",
        "contexts",
        "history",
        "layers",
        "transformer_layers",
        "transformer_heads",
        "feedforward_width",
        "conv_filters",
        "conv_kernel",
        "ema",
        "epsilon",
        "dropout",
        "lr",
        "batch_size",
        "l2",
        "negative_tails",
        "gradient_clip",
        "max_epochs",
        "patience",
        "inverse_relations",
        "batchnorm",
        "gate",
        "encoder_swap",
        "objective",
        "weight_decay",
        "transformer_initialization",
        "relation_initialization",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
    for key, default, choices in [
        (
            "transformer_initialization",
            "cloned_default",
            {"cloned_default", "independent"},
        ),
        (
            "relation_initialization",
            "tensor_xavier",
            {"tensor_xavier", "per_relation_xavier"},
        ),
    ]:
        if config.get(key, default) not in choices:
            raise ValueError(f"Unsupported {key}: {config.get(key)!r}")
    gate, swap = {
        "HA-AFSNet": ("adaptive", False),
        "HA-static": ("static", False),
        "HA-swap": ("adaptive", True),
    }[model]
    fixed = {
        "contexts": 5,
        "ema": 0.9,
        "epsilon": 1e-8,
        "negative_tails": 1,
        "inverse_relations": False,
        "batchnorm": False,
        "objective": "sampled_bce_sum_with_l2",
        "gate": gate,
        "encoder_swap": swap,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Unsupported {model} configuration {key}: expected {expected!r}"
            )
    if any(config.get(key, 0) < 1 for key in ["history", "max_epochs", "patience"]):
        raise ValueError("History, epochs and patience must be positive")
    if not 0 <= config.get("dropout", -1) < 1:
        raise ValueError("Invalid dropout")
    if config.get("model", model) != model:
        raise ValueError("Configuration model does not match the requested model")
    for key in [
        "hidden",
        "attention_width",
        "context_width",
        "activity_width",
        "composition_width",
        "situation_width",
        "history",
        "layers",
        "transformer_layers",
        "transformer_heads",
        "feedforward_width",
        "conv_filters",
        "conv_kernel",
        "batch_size",
        "max_epochs",
        "patience",
    ]:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["hidden"] % config["transformer_heads"]:
        raise ValueError("hidden must be divisible by transformer_heads")
    if config["conv_kernel"] % 2 == 0:
        raise ValueError("conv_kernel must be odd to preserve the embedding width")
    for key in ["lr", "gradient_clip"]:
        if not isinstance(config.get(key), (int, float)) or not 0 < config[key] < float(
            "inf"
        ):
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(config.get("l2"), (int, float)) or not 0 <= config["l2"] < float(
        "inf"
    ):
        raise ValueError("l2 must be finite and nonnegative")
    if config.get("weight_decay", 0) != 0:
        raise ValueError("Use explicit l2; additional Adam weight_decay is unsupported")
