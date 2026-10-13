"""Serial complete-epoch/full-test measurement from a selected checkpoint.

Concurrent training timings are retained separately and never used for this comparison.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from ha_afsnet.checkpoint import read_checkpoint, restore_rng, rng_state
from ha_afsnet.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
    canonical_config,
    canonical_protocol,
    validate_config,
    validate_protocol,
)
from ha_afsnet.config import (
    PROJECT_ROOT as ROOT,
)
from ha_afsnet.data import RegionalData
from ha_afsnet.evaluation import evaluate
from ha_afsnet.task import ForecastingTask
from ha_afsnet.utils import (
    artifact_reference,
    dump,
    resource_snapshot,
    seed_everything,
    sha256,
    synchronize,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.inference_repeats < 1:
        raise ValueError("Inference repeats must be positive")
    loaded, checkpoint_sha = read_checkpoint(args.checkpoint)
    model, region, seed = loaded["model"], loaded["dataset"], loaded["seed"]
    config = canonical_config(loaded["config"])
    protocol = canonical_protocol(loaded["protocol"])
    if model not in ["HA-AFSNet", "HA-static", "HA-swap"]:
        raise ValueError("This project measures HA-AFSNet and its two controls")
    validate_config(model, config)
    validate_protocol(protocol)
    seed_everything(seed)
    data = RegionalData(region, args.device, args.data_root, args.prepared_root)
    data_matches = (
        loaded["input_fingerprint"] == data.input_fingerprint
        if "input_fingerprint" in loaded
        else loaded["data_manifest_sha256"] in data.compatible_manifest_hashes
    )
    if not data_matches:
        raise ValueError("Checkpoint data does not match the measurement dataset")
    adapter = ForecastingTask(data, config)
    adapter.model.load_state_dict(loaded["state_dict"])
    del loaded
    if data.device.type == "cuda":
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        others = [
            int(pid.strip())
            for pid in processes
            if pid.strip().isdigit() and int(pid.strip()) != os.getpid()
        ]
        if others:
            raise RuntimeError(
                f"Serial measurement requires other CUDA jobs to finish; active PIDs: {others}"
            )
    dates = [
        t
        for t in data.splits["train"]
        if t >= (config["history"] * 2 if model.startswith("HA-") else 1)
    ]
    if not dates:
        raise ValueError("history leaves no usable training dates")
    selected = {
        k: v.detach().cpu().clone() for k, v in adapter.model.state_dict().items()
    }
    rng = rng_state()
    optimizer = torch.optim.Adam(
        adapter.model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0),
    )
    adapter.model.train()
    rows = data.rows[dates[0]]
    adapter.loss(dates[0], rows[: config["batch_size"]]).backward()
    optimizer.zero_grad(set_to_none=True)
    adapter.model.load_state_dict(selected)
    restore_rng(rng)
    if data.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(data.device)
    synchronize(data.device)
    tick = time.perf_counter()
    losses = []
    for t in dates:
        iterator = adapter.training_batches(t, data.rows[t])
        for loss in iterator:
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite serial benchmark loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                adapter.model.parameters(),
                config["gradient_clip"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
    synchronize(data.device)
    training_seconds = time.perf_counter() - tick
    training_peak = resource_snapshot(data.device)
    del optimizer
    adapter.model.load_state_dict(selected)
    del selected
    if data.device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate(adapter, data, "test", protocol["tie_handling"], 3)
    timings = []
    peaks = []
    for repetition in range(args.inference_repeats):
        if data.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(data.device)
        synchronize(data.device)
        tick = time.perf_counter()
        metrics = evaluate(adapter, data, "test", protocol["tie_handling"])
        synchronize(data.device)
        timings.append(time.perf_counter() - tick)
        peaks.append(resource_snapshot(data.device))
    if checkpoint_sha != sha256(args.checkpoint):
        raise RuntimeError("Input checkpoint changed during measurement")
    folder = args.output_root / f"resources_serial/{model}/{region}/seed_{seed}"
    result = {
        "status": "completed_serial_resource_benchmark",
        "model": model,
        "dataset": region,
        "seed": seed,
        "hidden_width": config["hidden"],
        "trainable_parameters": sum(
            p.numel() for p in adapter.model.parameters() if p.requires_grad
        ),
        "checkpoint": artifact_reference(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "training_seconds_per_epoch": [training_seconds],
        "training_peaks": [training_peak],
        "inference_seconds": timings,
        "inference_peaks": peaks,
        "metrics_at_selected_checkpoint": metrics,
        "device": str(data.device),
        "exclusive_gpu": data.device.type == "cuda",
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
        ).stdout
        if data.device.type == "cuda"
        else "CPU",
        "protocol": {
            "execution": "serial; no other CUDA processes admitted",
            "training_measurements_per_seed": 1,
            "training_scope": "one complete representative training epoch initialized from best validation checkpoint; disposable Adam and weights",
            "training_warmup": "one forward/backward, then restore model and all RNG; exclude warmup",
            "training_includes": "causal graph construction/transfer, encoding, scoring, loss, backward, clipping, Adam",
            "training_excludes": "initial data/statistics preparation, context fitting, validation, checkpoint and logging I/O",
            "inference_warmup": "three complete dates before timed full test",
            "inference_repeats": args.inference_repeats,
            "inference_includes": "all source test rows, graph construction, encoding, full vocabulary scoring, popularity and filtering",
            "inference_excludes": "initial preparation and checkpoint/rank archive I/O",
            "synchronize": "before/after every timed region",
            "memory": "PyTorch allocator peak allocated and reserved separately, reset per phase; restoration weights retained on CPU",
            "summary": "arithmetic mean, sample SD, minimum and maximum; preserve per-seed measurements",
        },
    }
    dump(folder / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
