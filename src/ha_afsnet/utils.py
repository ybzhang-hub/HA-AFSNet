"""Random seeds, portable metadata and synchronized resource measurement."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT

MANIFEST_FIELDS = [
    "dataset",
    "entities",
    "relations",
    "days",
    "date_min",
    "date_max",
    "splits",
    "source_sha256",
    "popularity",
]


def manifest_fingerprint(manifest):
    """Fingerprint split contents, vocabulary and popularity settings."""
    contents = {key: manifest.get(key) for key in MANIFEST_FIELDS}
    payload = json.dumps(contents, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def artifact_reference(path):
    """Return a portable reference for an output artifact."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return Path(path).name


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(4)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resource_snapshot(device):
    if device.type != "cuda":
        return {"allocated_GiB": 0, "reserved_GiB": 0}
    return {
        "allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
        "reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30,
    }
