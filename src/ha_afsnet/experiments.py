"""Run independent regions and random seeds with bounded process concurrency."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
    PROJECT_ROOT,
    canonical_config,
)
from .trainer import training_source_hashes
from .utils import dump, manifest_fingerprint, sha256


def validate_completed_run(
    destination, job, protocol, config_path=None, validation_only=False
):
    result = json.loads((destination / "result.json").read_text())
    model, region, seed = job
    name = {
        "HA-AFSNet": "ha_afsnet",
        "HA-static": "static_gate",
        "HA-swap": "encoder_swap",
    }[model]
    expected = canonical_config(
        json.loads((config_path or PROJECT_ROOT / f"configs/{name}.json").read_text())
    )
    configuration = canonical_config(
        json.loads((destination / "config.json").read_text())
    )
    reference = json.loads((PROJECT_ROOT / f"splits/{region}.json").read_text())
    matched_data = (
        result["input_fingerprint"] == manifest_fingerprint(reference)
        if "input_fingerprint" in result
        else result["data_manifest_sha256"]
        == reference.get("reference_manifest_sha256")
    )
    if (
        result["status"] != ("completed_validation" if validation_only else "completed")
        or result["full_test"] == validation_only
        or result.get("smoke")
        or (result["model"], result["dataset"], result["seed"]) != job
        or result["protocol_sha256"] != sha256(protocol)
        or configuration != expected
        or not matched_data
        or result.get("training_source_sha256") != training_source_hashes(model)
        or result["checkpoint_sha256"] != sha256(destination / "best.pt")
    ):
        raise ValueError(f"Completed run differs from the requested experiment: {job}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["HA-AFSNet", "HA-static", "HA-swap"],
        default=["HA-AFSNet", "HA-static", "HA-swap"],
    )
    parser.add_argument(
        "--regions", nargs="+", choices=["EG", "IR", "IS"], default=["EG", "IR", "IS"]
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=[42, 43, 44, 45, 46],
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--diagnostics-every-steps", type=int, default=100)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--prepared-root", type=str, default=str(DEFAULT_PREPARED_ROOT))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.data_root = str(Path(args.data_root).resolve())
    args.prepared_root = str(Path(args.prepared_root).resolve())
    if args.config is not None:
        args.config = args.config.resolve()
        if len(args.models) != 1:
            raise ValueError("A custom configuration requires exactly one model")
        if not args.config.is_file():
            raise FileNotFoundError(args.config)
    if args.diagnostics_every_steps < 0:
        raise ValueError("Diagnostics interval must be nonnegative")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    for values in [args.models, args.regions, args.seeds]:
        if len(values) != len(set(values)):
            raise ValueError("Models, regions and seeds must be unique")
    protocol = PROJECT_ROOT / "configs/evaluation.json"
    pending = list(itertools.product(args.models, args.regions, args.seeds))
    active, records = {}, {}
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    def folder(job):
        model, region, seed = job
        return args.output_root / f"runs/{model}/{region}/seed_{seed}"

    while pending or active:
        for job, (process, stream) in list(active.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            stream.close()
            del active[job]
            if exit_code:
                records[job].update(status="failed", exit_code=exit_code)
                dump(
                    logs / "experiment_status.json",
                    {"status": "failed", "jobs": list(records.values())},
                )
                raise RuntimeError(
                    f"Experiment failed: {job}; inspect its log before resuming the queue"
                )
            validate_completed_run(
                folder(job), job, protocol, args.config, args.validation_only
            )
            records[job].update(status="completed")
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/summarize.py"),
                    "--output-root",
                    str(args.output_root),
                ],
                check=True,
            )
        while pending and len(active) < args.workers:
            job = pending.pop(0)
            destination = folder(job)
            result_path = destination / "result.json"
            if result_path.exists():
                validate_completed_run(
                    destination, job, protocol, args.config, args.validation_only
                )
                records[job] = dict(
                    model=job[0], dataset=job[1], seed=job[2], status="completed"
                )
                continue
            command = [
                sys.executable,
                str(PROJECT_ROOT / "train.py"),
                "--model",
                job[0],
                "--dataset",
                job[1],
                "--seed",
                str(job[2]),
                "--device",
                args.device,
                "--data-root",
                args.data_root,
                "--prepared-root",
                args.prepared_root,
                "--output-root",
                str(args.output_root),
                "--protocol",
                str(protocol),
                "--execution-mode",
                "parallel",
                "--diagnostics-every-steps",
                str(args.diagnostics_every_steps),
            ]
            if args.config is not None:
                command.extend(["--config", str(args.config)])
            if args.validation_only:
                command.append("--validation-only")
            if (destination / "last.pt").exists():
                command.append("--resume")
            stream = (logs / f"{job[0]}_{job[1]}_{job[2]}.log").open("a")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[job] = process, stream
            records[job] = dict(
                model=job[0], dataset=job[1], seed=job[2], status="running"
            )
        dump(
            logs / "experiment_status.json",
            {
                "status": "running",
                "workers": args.workers,
                "active": len(active),
                "queued": len(pending),
                "jobs": list(records.values()),
            },
        )
        if active:
            time.sleep(5)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/summarize.py"),
            "--output-root",
            str(args.output_root),
        ],
        check=True,
    )
    dump(
        logs / "experiment_status.json",
        {"status": "completed", "jobs": list(records.values())},
    )


if __name__ == "__main__":
    main()
