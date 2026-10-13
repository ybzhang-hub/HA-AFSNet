"""Summarize completed experiments and compute matched-seed confidence intervals."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from ha_afsnet.config import PROJECT_ROOT as ROOT
from ha_afsnet.utils import dump

METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
SEEDS = [42, 43, 44, 45, 46]


def write_csv(path, rows, fields):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize completed per-seed experiments and matched comparisons"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    out = output_root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    runs, directories = [], {}
    for path in (output_root / "runs").glob("*/*/seed_*/result.json"):
        record = json.loads(path.read_text())
        if (
            record.get("status") == "completed"
            and record.get("full_test")
            and not record.get("smoke")
        ):
            record["config"] = str(
                (path.parent / "config.json").relative_to(output_root)
            )
            record["checkpoint"] = str(
                (path.parent / "best.pt").relative_to(output_root)
            )
            directories[(record["model"], record["dataset"], record["seed"])] = (
                path.parent
            )
            runs.append(record)
    runs = [
        r
        for r in runs
        if r.get("status") == "completed" and r.get("full_test") and not r.get("smoke")
    ]
    per_seed, grouped, resources, observed_resources = [], {}, [], []
    for r in runs:
        per_seed.append(r)
        group = grouped.setdefault((r["model"], r["dataset"]), {})
        if r["seed"] in group:
            raise ValueError("Duplicate model/dataset/seed result")
        group[r["seed"]] = r
        directory = directories[(r["model"], r["dataset"], r["seed"])]
        configuration = json.loads((directory / "config.json").read_text())
        epochs = (
            json.loads((directory / "epochs.json").read_text())
            if (directory / "epochs.json").exists()
            else []
        )
        for kind, values in [
            ("training_seconds_per_epoch", [x["training_seconds"] for x in epochs]),
            ("full_test_seconds", r["inference_seconds"]),
            (
                "training_peak_allocated_GiB",
                [x["train_peak_allocated_GiB"] for x in epochs],
            ),
            (
                "training_peak_reserved_GiB",
                [x["train_peak_reserved_GiB"] for x in epochs],
            ),
            (
                "inference_peak_allocated_GiB",
                [x["allocated_GiB"] for x in r["inference_peaks"]],
            ),
            (
                "inference_peak_reserved_GiB",
                [x["reserved_GiB"] for x in r["inference_peaks"]],
            ),
        ]:
            if values:
                item = {
                    "model": r["model"],
                    "dataset": r["dataset"],
                    "seed": r["seed"],
                    "measurement": kind,
                    "hidden_width": configuration["hidden"],
                    "trainable_parameters": r["trainable_parameters"],
                    "hardware_and_protocol_record": str(
                        (directory / "result.json").relative_to(output_root)
                    ),
                    "exclusive_gpu": r["measurement_protocol"]["exclusive_gpu"],
                    **describe(values),
                }
                observed_resources.append(item)
                if (
                    r["measurement_protocol"].get("resource_comparison_eligible", True)
                    and r["measurement_protocol"]["exclusive_gpu"]
                ):
                    resources.append(item)
    for path in (output_root / "resources_serial").glob("*/*/seed_*/result.json"):
        record = json.loads(path.read_text())
        if record["status"] != "completed_serial_resource_benchmark":
            continue
        for kind, values in [
            ("training_seconds_per_epoch", record["training_seconds_per_epoch"]),
            ("full_test_seconds", record["inference_seconds"]),
            (
                "training_peak_allocated_GiB",
                [v["allocated_GiB"] for v in record["training_peaks"]],
            ),
            (
                "training_peak_reserved_GiB",
                [v["reserved_GiB"] for v in record["training_peaks"]],
            ),
            (
                "inference_peak_allocated_GiB",
                [v["allocated_GiB"] for v in record["inference_peaks"]],
            ),
            (
                "inference_peak_reserved_GiB",
                [v["reserved_GiB"] for v in record["inference_peaks"]],
            ),
        ]:
            resources.append(
                {
                    "model": record["model"],
                    "dataset": record["dataset"],
                    "seed": record["seed"],
                    "measurement": kind,
                    "hidden_width": record["hidden_width"],
                    "trainable_parameters": record["trainable_parameters"],
                    "hardware_and_protocol_record": str(path.relative_to(output_root)),
                    "exclusive_gpu": record.get("exclusive_gpu", True),
                    **describe(values),
                }
            )
    means, differences = [], []
    for (model, dataset), seeds in grouped.items():
        configurations = {
            json.dumps(
                json.loads(
                    (
                        directories[(r["model"], r["dataset"], r["seed"])]
                        / "config.json"
                    ).read_text()
                ),
                sort_keys=True,
            )
            for r in seeds.values()
        }
        if len(configurations) != 1 or any(
            len({r[key] for r in seeds.values()}) != 1
            for key in ["protocol_sha256", "data_manifest_sha256", "evaluation_sha256"]
        ):
            raise ValueError(
                f"Cannot aggregate different configurations/protocols/splits/evaluators as repeated seeds: {model}/{dataset}"
            )
        for metric in METRICS:
            stats = describe([r[metric] for r in seeds.values()])
            half = None
            if stats["n"] > 1:
                half = (
                    student_t.ppf(0.975, stats["n"] - 1)
                    * stats["sd"]
                    / np.sqrt(stats["n"])
                )
            means.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": metric,
                    **stats,
                    "ci95_low": stats["mean"] - half if half is not None else None,
                    "ci95_high": stats["mean"] + half if half is not None else None,
                    "complete_five_seeds": set(seeds) == set(SEEDS),
                }
            )
    for dataset in ["EG", "IR", "IS"]:
        ours = grouped.get(("HA-AFSNet", dataset), {})
        for comparator in ["HA-static", "HA-swap"]:
            theirs = grouped.get((comparator, dataset), {})
            if set(ours) != set(SEEDS) or set(theirs) != set(SEEDS):
                continue
            if not all(
                ours[s]["protocol_sha256"] == theirs[s]["protocol_sha256"]
                and ours[s]["data_manifest_sha256"] == theirs[s]["data_manifest_sha256"]
                and ours[s]["evaluation_sha256"] == theirs[s]["evaluation_sha256"]
                for s in SEEDS
            ):
                raise ValueError(
                    "Paired comparison requires matched protocols/splits/evaluators"
                )
            for metric in METRICS:
                delta = np.array([ours[s][metric] - theirs[s][metric] for s in SEEDS])
                stats = describe(delta)
                half = student_t.ppf(0.975, 4) * stats["sd"] / np.sqrt(5)
                statistic = abs(delta.mean())
                null = np.array(
                    [
                        abs((delta * signs).mean())
                        for signs in itertools.product([-1, 1], repeat=5)
                    ]
                )
                p = float(np.mean(null >= statistic - 1e-15))
                differences.append(
                    {
                        "dataset": dataset,
                        "comparator": comparator,
                        "metric": metric,
                        **{
                            f"delta_seed_{s}": float(value)
                            for s, value in zip(SEEDS, delta)
                        },
                        **stats,
                        "ci95_low": stats["mean"] - half,
                        "ci95_high": stats["mean"] + half,
                        "exact_two_sided_sign_flip_p": p,
                    }
                )
    write_csv(
        out / "per_seed_results.csv",
        per_seed,
        [
            "model",
            "dataset",
            "seed",
            *METRICS,
            "best_epoch",
            "validation_MRR",
            "retained_queries",
            "config",
            "checkpoint",
            "checkpoint_sha256",
            "protocol_sha256",
            "data_manifest_sha256",
            "evaluation_sha256",
        ],
    )
    write_csv(
        out / "model_summary.csv",
        means,
        [
            "model",
            "dataset",
            "metric",
            "n",
            "mean",
            "sd",
            "minimum",
            "maximum",
            "ci95_low",
            "ci95_high",
            "complete_five_seeds",
        ],
    )
    write_csv(
        out / "paired_differences.csv",
        differences,
        [
            "dataset",
            "comparator",
            "metric",
            *[f"delta_seed_{s}" for s in SEEDS],
            "n",
            "mean",
            "sd",
            "minimum",
            "maximum",
            "ci95_low",
            "ci95_high",
            "exact_two_sided_sign_flip_p",
        ],
    )
    write_csv(
        out / "resources.csv",
        resources,
        [
            "model",
            "dataset",
            "seed",
            "hidden_width",
            "trainable_parameters",
            "measurement",
            "n",
            "mean",
            "sd",
            "minimum",
            "maximum",
            "exclusive_gpu",
            "hardware_and_protocol_record",
        ],
    )
    write_csv(
        out / "resources_observed_training.csv",
        observed_resources,
        [
            "model",
            "dataset",
            "seed",
            "hidden_width",
            "trainable_parameters",
            "measurement",
            "n",
            "mean",
            "sd",
            "minimum",
            "maximum",
            "exclusive_gpu",
            "hardware_and_protocol_record",
        ],
    )
    plan = []
    for model, dataset, seed in itertools.product(
        ["HA-AFSNet", "HA-static", "HA-swap"], ["EG", "IR", "IS"], SEEDS
    ):
        running_path = output_root / f"runs/{model}/{dataset}/seed_{seed}/status.json"
        current = (
            json.loads(running_path.read_text()).get("status")
            if running_path.exists()
            else None
        )
        status = (
            "completed"
            if seed in grouped.get((model, dataset), {})
            else current or "not_completed"
        )
        plan.append(
            {"model": model, "dataset": dataset, "seed": seed, "status": status}
        )
    write_csv(
        out / "experiment_status.csv", plan, ["model", "dataset", "seed", "status"]
    )
    dump(
        out / "summary_status.json",
        {
            "completed_full_runs": len(runs),
            "paired_comparisons": len(differences),
            "expected_HA_main_and_controls": 45,
        },
    )
    print(
        json.dumps(
            {"completed_full_runs": len(runs), "paired_comparisons": len(differences)}
        )
    )


if __name__ == "__main__":
    main()
