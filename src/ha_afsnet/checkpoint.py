"""Atomic checkpoints and full random-state restoration."""

import hashlib
import os
import random

import numpy as np
import torch


def read_checkpoint(path):
    """Load and fingerprint the same file snapshot, including during atomic replacement."""
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
    return payload, digest.hexdigest()


def atomic_torch_save(payload, path):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
