"""Causal training, validation-selected checkpoints, ranking and gate export."""

import argparse
import fcntl
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .checkpoint import atomic_torch_save, read_checkpoint, restore_rng, rng_state
from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
    canonical_config,
    canonical_protocol,
    validate_config,
    validate_protocol,
)
from .config import (
    PROJECT_ROOT as ROOT,
)
from .data import RegionalData
from .diagnostics import TrainingDiagnostics, gradient_groups
from .evaluation import evaluate
from .task import ForecastingTask
from .utils import (
    artifact_reference,
    dump,
    manifest_fingerprint,
    resource_snapshot,
    seed_everything,
    sha256,
    synchronize,
)

RUN_STATUS_PATH = None
RUN_LOCK = None


def training_source_hashes(model):
    paths = sorted((ROOT / "src/ha_afsnet").rglob("*.py"))
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
        if "__pycache__" not in path.parts
    }


def _run():
    global RUN_STATUS_PATH, RUN_LOCK
    RUN_STATUS_PATH = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["HA-AFSNet", "HA-static", "HA-swap"], default="HA-AFSNet"
    )
    parser.add_argument("--dataset", choices=["EG", "IR", "IS"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44, 45, 46], default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/evaluation.json"
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Short execution check; excluded from experiment summaries",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Stop after validation-selected training; do not evaluate test during tuning",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore an interrupted training run, including optimizer and all RNG states",
    )
    parser.add_argument("--save-every-dates", type=int, default=50)
    parser.add_argument(
        "--diagnostics-every-steps",
        type=int,
        default=100,
        help="Sample loss, score, gate and module-gradient signals; 0 disables diagnostics",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--execution-mode", choices=["serial", "parallel"], default="serial"
    )
    args = parser.parse_args()
    if args.quick_run and args.epochs is not None:
        raise ValueError("--quick-run fixes epochs=1; do not also pass --epochs")
    if args.resume and args.eval_only:
        raise ValueError("--resume is for training")
    if args.validation_only and (args.eval_only or args.quick_run):
        raise ValueError("--validation-only requires full training")
    if args.save_every_dates < 1:
        raise ValueError("Checkpoint interval must be positive")
    if args.diagnostics_every_steps < 0:
        raise ValueError("Diagnostics interval must be nonnegative")
    if args.eval_only and args.epochs is not None:
        raise ValueError("--epochs does not apply to checkpoint evaluation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    protocol = canonical_protocol(json.loads(args.protocol.read_text()))
    validate_protocol(protocol)
    name = {
        "HA-AFSNet": "ha_afsnet",
        "HA-static": "static_gate",
        "HA-swap": "encoder_swap",
    }[args.model]
    training_directory = (
        args.output_root
        / ("smoke" if args.quick_run else "runs")
        / args.model
        / args.dataset
        / f"seed_{args.seed}"
    )
    checkpoint = args.checkpoint or training_directory / "best.pt"
    evaluation_checkpoint = None
    if args.eval_only:
        evaluation_checkpoint, checkpoint_fingerprint = read_checkpoint(checkpoint)
        if any(
            evaluation_checkpoint[key] != value
            for key, value in [
                ("model", args.model),
                ("dataset", args.dataset),
                ("seed", args.seed),
            ]
        ):
            raise ValueError("Checkpoint seed/dataset/model mismatch")
        config = (
            dict(evaluation_checkpoint["config"])
            if args.config is None
            else json.loads(args.config.read_text())
        )
    else:
        config = json.loads((args.config or ROOT / f"configs/{name}.json").read_text())
    config = canonical_config(config)
    if args.epochs is not None:
        config["max_epochs"] = args.epochs
    if args.quick_run and not args.eval_only:
        config["max_epochs"] = 1
    validate_config(args.model, config)
    if args.inference_repeats < 1:
        raise ValueError("Inference repeats must be positive")
    seed_everything(args.seed)
    data_start = time.perf_counter()
    data = RegionalData(args.dataset, args.device, args.data_root, args.prepared_root)
    adapter = ForecastingTask(data, config)
    synchronize(data.device)
    preprocessing_seconds = time.perf_counter() - data_start
    if args.eval_only:
        evaluation_id = (
            checkpoint_fingerprint[:16]
            + "_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        destination = (
            args.output_root
            / "evaluation"
            / args.model
            / args.dataset
            / f"seed_{args.seed}"
            / evaluation_id
        )
    else:
        destination = training_directory
        if (destination / "result.json").exists():
            raise FileExistsError(
                f"Completed training already exists at {destination}; evaluate its checkpoint or use scripts/run_experiments.py to skip completed runs"
            )
        if args.checkpoint is not None:
            raise ValueError(
                "--checkpoint is an input to --eval-only; training writes best.pt in its run directory"
            )
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / ".run.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "This experiment directory already has an active process"
        ) from None
    RUN_LOCK = lock
    if not args.eval_only and (destination / "result.json").exists():
        raise FileExistsError("Completed training already exists")
    last_checkpoint = destination / "last.pt"
    if not args.eval_only and last_checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"Interrupted training exists at {destination}; use --resume"
        )
    if args.resume and not last_checkpoint.exists():
        raise FileNotFoundError(f"No resumable checkpoint at {last_checkpoint}")
    gpu_state = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
        ).stdout
        if data.device.type == "cuda"
        else "CPU"
    )
    info = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "status": "running",
        "smoke": args.quick_run,
        "evaluation_only": args.eval_only,
        "validation_only": args.validation_only,
        "training_directory": artifact_reference(
            checkpoint.parent if args.eval_only else destination
        ),
        "config": artifact_reference(destination / "config.json"),
        "checkpoint": artifact_reference(checkpoint),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_state_at_start": gpu_state,
        "preprocessing_seconds": preprocessing_seconds,
        "trainable_parameters": sum(
            p.numel() for p in adapter.model.parameters() if p.requires_grad
        ),
        "data_manifest_sha256": sha256(
            data.prepared_root / args.dataset / "manifest.json"
        ),
        "input_fingerprint": manifest_fingerprint(
            json.loads(
                (data.prepared_root / args.dataset / "manifest.json").read_text()
            )
        ),
        "evaluation_sha256": sha256(ROOT / "src/ha_afsnet/evaluation.py"),
        "protocol_sha256": sha256(args.protocol),
        "training_source_sha256": training_source_hashes(args.model),
        "source_sha256": training_source_hashes(args.model),
        "data_input": "external released dataset",
        "execution_mode": args.execution_mode,
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "diagnostics_every_steps": args.diagnostics_every_steps,
        "measurement_protocol": {
            "cuda_synchronize": "before and after every timed region",
            "warmup": "one training forward/backward before timing with state and RNG restored; three evaluation dates before timing",
            "training_includes": "history graph construction, transfer, encoding, scoring, loss, backward, clipping and optimizer",
            "training_excludes": "archive extraction, initial text/context fitting, initial data load, validation, checkpoint I/O",
            "preprocessing_seconds_includes": "initial data load, causal calendar-statistic precomputation and model construction",
            "inference_includes": "history graph construction, transfer, encoding, all entity scoring, popularity selection and filtered ranking",
            "inference_excludes": "initial data/context fitting, checkpoint load, optimizer state and rank archive I/O",
            "memory": "PyTorch CUDA allocator peak allocated and reserved, reset before each phase; includes resident weights and tensor graph cache; excludes CUDA context and allocations outside the PyTorch allocator",
            "repeats": 1 if args.quick_run else args.inference_repeats,
            "summary": "arithmetic mean, sample SD and min/max; each seed retained",
            "exclusive_gpu": False,
            "resource_comparison_eligible": args.execution_mode == "serial",
        },
    }
    optimizer = torch.optim.Adam(
        adapter.model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )
    best, best_epoch, patience, epoch_rows = -float("inf"), 0, 0, []
    resume_epoch, resume_progress = 1, None
    optimizer_steps = 0
    epoch_diagnostics = None
    if args.resume:
        restored = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        for key in [
            "model",
            "dataset",
            "seed",
            "config",
            "protocol",
            "data_manifest_sha256",
            "training_source_sha256",
        ]:
            expected = (
                config
                if key == "config"
                else protocol
                if key == "protocol"
                else info[key]
            )
            if restored[key] != expected:
                raise ValueError(f"Resume provenance mismatch: {key}")
        if restored.get("diagnostics_every_steps", 0) != args.diagnostics_every_steps:
            raise ValueError("Resume diagnostics interval mismatch")
        if restored.get("validation_only", False) != args.validation_only:
            raise ValueError("Resume validation-only mode mismatch")
        adapter.model.load_state_dict(restored["state_dict"])
        optimizer.load_state_dict(restored["optimizer"])
        restore_rng(restored["rng"])
        best, best_epoch, patience = (
            restored["best"],
            restored["best_epoch"],
            restored["patience"],
        )
        epoch_rows, resume_epoch, resume_progress = (
            restored["epoch_rows"],
            restored["next_epoch"],
            restored["progress"],
        )
        optimizer_steps = restored.get("optimizer_steps", 0)
        dump(destination / "epochs.json", epoch_rows)
        info["resumed_from"] = artifact_reference(last_checkpoint)
        del restored
    RUN_STATUS_PATH = destination / "status.json"
    dump(destination / "config.json", config)
    dump(destination / "protocol.json", protocol)
    dump(destination / "status.json", info)

    def save_training(next_epoch, progress):
        atomic_torch_save(
            {
                "model": args.model,
                "dataset": args.dataset,
                "seed": args.seed,
                "config": config,
                "protocol": protocol,
                "data_manifest_sha256": info["data_manifest_sha256"],
                "input_fingerprint": info["input_fingerprint"],
                "training_source_sha256": info["training_source_sha256"],
                "state_dict": adapter.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "best": best,
                "best_epoch": best_epoch,
                "patience": patience,
                "epoch_rows": epoch_rows,
                "next_epoch": next_epoch,
                "progress": progress,
                "optimizer_steps": optimizer_steps,
                "diagnostics_every_steps": args.diagnostics_every_steps,
                "validation_only": args.validation_only,
            },
            last_checkpoint,
        )

    train_dates = data.splits["train"]
    start = config["history"] * 2
    train_dates = [t for t in train_dates if t >= start]
    if not train_dates:
        raise ValueError("history leaves no usable training dates")
    if args.quick_run:
        train_dates = train_dates[:3]
    eval_limit = 2 if args.quick_run else 0
    query_limit = 128 if args.quick_run else 0
    if not args.eval_only:
        state = {
            k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()
        }
        torch_rng, cuda_rng, numpy_rng, python_rng = (
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
            np.random.get_state(),
            random.getstate(),
        )
        adapter.model.train()
        warmup_limit = config["batch_size"]
        adapter.loss(
            train_dates[0], data.rows[train_dates[0]][:warmup_limit]
        ).backward()
        optimizer.zero_grad(set_to_none=True)
        adapter.model.load_state_dict(state)
        del state
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        synchronize(data.device)
        for epoch in range(
            resume_epoch,
            config["max_epochs"] + 1 if patience < config["patience"] else resume_epoch,
        ):
            adapter.model.train()
            if data.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(data.device)
            synchronize(data.device)
            tick = time.perf_counter()
            if resume_progress is not None:
                diagnostic_state = resume_progress.get("diagnostics")
                dates, losses = resume_progress["dates"], resume_progress["losses"]
                completed_dates, elapsed_before = (
                    resume_progress["completed_dates"],
                    resume_progress["training_seconds"],
                )
                peak_before = resume_progress["peak_memory"]
                resume_progress = None
            else:
                diagnostic_state = None
                dates, losses, completed_dates, elapsed_before, peak_before = (
                    list(train_dates),
                    [],
                    0,
                    0.0,
                    {},
                )
            epoch_diagnostics = (
                TrainingDiagnostics(
                    data.device, config["gradient_clip"], diagnostic_state
                )
                if args.diagnostics_every_steps
                else None
            )
            for i in range(completed_dates, len(dates)):
                t = dates[i]
                rows = data.rows[t][: query_limit or None]
                iterator = adapter.training_batches(
                    t, rows, args.diagnostics_every_steps, optimizer_steps
                )
                for batch_value in iterator:
                    loss, details = (
                        batch_value
                        if args.diagnostics_every_steps
                        else (batch_value, None)
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Nonfinite loss at epoch={epoch}, date={t}"
                        )
                    loss.backward()
                    if details is not None:
                        details.update(gradient_groups(adapter.model))
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        adapter.model.parameters(),
                        config["gradient_clip"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if epoch_diagnostics is not None:
                        epoch_diagnostics.observe(gradient_norm, details)
                    losses.append(float(loss.detach()))
                if (i + 1) % args.save_every_dates == 0:
                    synchronize(data.device)
                    elapsed_before += time.perf_counter() - tick
                    peak_before = {
                        k: max(v, peak_before.get(k, 0))
                        for k, v in resource_snapshot(data.device).items()
                    }
                    save_training(
                        epoch,
                        {
                            "dates": dates,
                            "losses": losses,
                            "completed_dates": i + 1,
                            "training_seconds": elapsed_before,
                            "peak_memory": peak_before,
                            "diagnostics": epoch_diagnostics.state_dict()
                            if epoch_diagnostics is not None
                            else None,
                        },
                    )
                    info.update(
                        epoch=epoch,
                        completed_dates=i + 1,
                        total_dates=len(dates),
                        training_seconds_so_far=elapsed_before,
                    )
                    dump(destination / "status.json", info)
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "date": t,
                                "completed_dates": i + 1,
                                "total_dates": len(dates),
                                "training_seconds_so_far": elapsed_before,
                                "loss": float(np.mean(losses[-50:])),
                            }
                        ),
                        flush=True,
                    )
                    synchronize(data.device)
                    tick = time.perf_counter()
            synchronize(data.device)
            elapsed = elapsed_before + time.perf_counter() - tick
            train_memory = resource_snapshot(data.device)
            train_memory = {
                k: max(v, peak_before.get(k, 0)) for k, v in train_memory.items()
            }
            validation = evaluate(
                adapter,
                data,
                "valid",
                protocol["tie_handling"],
                eval_limit,
                query_limit,
            )
            value = validation["MRR"]
            if value >= best:
                best, best_epoch, patience = value, epoch, 0
                atomic_torch_save(
                    {
                        "state_dict": adapter.model.state_dict(),
                        "epoch": epoch,
                        "validation_MRR": best,
                        "seed": args.seed,
                        "dataset": args.dataset,
                        "model": args.model,
                        "config": config,
                        "protocol": protocol,
                        "data_manifest_sha256": info["data_manifest_sha256"],
                        "input_fingerprint": info["input_fingerprint"],
                    },
                    checkpoint,
                )
            else:
                patience += 1
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "training_seconds": elapsed,
                "validation_MRR": value,
                "best_epoch": best_epoch,
                "best_validation_MRR": best,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps_total": optimizer_steps,
                "patience_used": patience,
                **{f"train_peak_{k}": v for k, v in train_memory.items()},
            }
            if epoch_diagnostics is not None:
                record["training_diagnostics"] = epoch_diagnostics.summary()
            epoch_rows.append(record)
            dump(destination / "epochs.json", epoch_rows)
            print(json.dumps(record), flush=True)
            save_training(epoch + 1, None)
            if patience >= config["patience"]:
                break
    optimizer = None
    if args.eval_only:
        loaded = evaluation_checkpoint
        evaluation_checkpoint = None
    else:
        loaded, checkpoint_fingerprint = read_checkpoint(checkpoint)
    if (
        loaded["seed"] != args.seed
        or loaded["dataset"] != args.dataset
        or loaded["model"] != args.model
    ):
        raise ValueError("Checkpoint seed/dataset/model mismatch")
    checkpoint_config = canonical_config(loaded["config"])
    data_matches = (
        loaded["input_fingerprint"] == info["input_fingerprint"]
        if "input_fingerprint" in loaded
        else loaded["data_manifest_sha256"]
        in getattr(data, "compatible_manifest_hashes", {info["data_manifest_sha256"]})
    )
    if (
        checkpoint_config != config
        or canonical_protocol(loaded["protocol"]) != protocol
        or not data_matches
    ):
        raise ValueError("Checkpoint configuration/protocol/data provenance mismatch")
    adapter.model.load_state_dict(loaded["state_dict"])
    best_epoch, best = loaded["epoch"], loaded["validation_MRR"]
    del loaded
    if args.validation_only:
        info.update(
            status="completed_validation",
            full_test=False,
            best_epoch=best_epoch,
            validation_MRR=best,
            checkpoint_sha256=checkpoint_fingerprint,
            epochs_completed=len(epoch_rows),
        )
        dump(destination / "result.json", info)
        dump(destination / "status.json", info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return
    if data.device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate(adapter, data, "test", protocol["tie_handling"], 3, query_limit)
    timings, memory_rows = [], []
    repeats = 1 if args.quick_run else args.inference_repeats
    rank_records = {}
    for repetition in range(repeats):
        if data.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(data.device)
        synchronize(data.device)
        tick = time.perf_counter()
        test = evaluate(
            adapter,
            data,
            "test",
            protocol["tie_handling"],
            eval_limit,
            query_limit,
            collector=rank_records if repetition == 0 else None,
        )
        synchronize(data.device)
        timings.append(time.perf_counter() - tick)
        memory_rows.append(resource_snapshot(data.device))
        if repetition == 0:
            np.savez_compressed(destination / "test_ranks.npz", **rank_records)
            rank_records.clear()
    if args.eval_only:
        epochs_path = checkpoint.parent / "epochs.json"
        epoch_rows = json.loads(epochs_path.read_text()) if epochs_path.exists() else []
    info.update(
        status="completed_evaluation"
        if args.eval_only
        else "completed_smoke"
        if args.quick_run
        else "completed",
        best_epoch=best_epoch,
        validation_MRR=best,
        checkpoint_sha256=checkpoint_fingerprint,
        **test,
        inference_seconds=timings,
        inference_peaks=memory_rows,
        epochs_completed=len(epoch_rows),
        full_test=not args.quick_run,
    )
    adapter.export_gates(destination / "gates.csv", args.seed)
    dump(destination / "result.json", info)
    dump(destination / "status.json", info)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def main():
    global RUN_LOCK
    try:
        _run()
    except Exception as error:
        if RUN_STATUS_PATH is not None:
            info = (
                json.loads(RUN_STATUS_PATH.read_text())
                if RUN_STATUS_PATH.exists()
                else {}
            )
            info.update(status="failed", error=f"{type(error).__name__}: {error}")
            dump(RUN_STATUS_PATH, info)
        raise
    finally:
        if RUN_LOCK is not None:
            RUN_LOCK.close()
            RUN_LOCK = None


if __name__ == "__main__":
    main()
