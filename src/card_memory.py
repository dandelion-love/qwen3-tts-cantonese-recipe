# SPDX-License-Identifier: Apache-2.0
"""Free memory on the card, and the floor check every GPU stage runs before it starts.

A failure to measure is not a measurement of low memory. The old probe ran nvidia-smi once and returned None on any
exception, and its callers refused on None exactly as on a low reading, so a single failed call could stop a speaker
phase after hours of language training.

free_mib() tries nvidia-smi PROBE_TRIES times with a short backoff, then torch.cuda.mem_get_info() in a SUBPROCESS (so
the calling process never holds a CUDA context), and returns the value WITH the method that produced it and every
method's error. check_floor() refuses only on a MEASURED value below the floor; when every method failed it prints a
loud WARNING and lets the stage proceed, and the caller writes the returned dict into its stage record.

    python src/test_card_memory.py      the CPU test, each case with faked probes
"""
from __future__ import annotations

import subprocess
import sys
import time

PROBE_TRIES = 3
PROBE_BACKOFF_S = 2.0
TORCH_PROBE = "import torch; f, _ = torch.cuda.mem_get_info(); print(f // (1 << 20))"


def nvidia_smi_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()[0]
    total, used = (int(x) for x in out.split(","))
    return total - used


def torch_mib(python: str | None = None) -> int:
    res = subprocess.run([python or sys.executable, "-c", TORCH_PROBE], capture_output=True, text=True, timeout=120)
    if res.returncode:
        raise RuntimeError(f"exit {res.returncode}: {(res.stderr.strip().splitlines() or ['no stderr'])[-1]}")
    return int(res.stdout.strip().splitlines()[-1])


def free_mib(python: str | None = None, tries: int = PROBE_TRIES, backoff_s: float = PROBE_BACKOFF_S) -> dict:
    """{"mib": int | None, "method": "nvidia-smi" | "torch" | None, "errors": [str]}; mib None means NOT MEASURED."""
    errors = []
    for i in range(tries):
        try:
            return {"mib": nvidia_smi_mib(), "method": "nvidia-smi", "errors": errors}
        except Exception as e:
            errors.append(f"nvidia-smi try {i + 1}: {type(e).__name__}: {e}")
            if i + 1 < tries:
                time.sleep(backoff_s * (i + 1))
    try:
        return {"mib": torch_mib(python), "method": "torch", "errors": errors}
    except Exception as e:
        errors.append(f"torch.cuda.mem_get_info: {type(e).__name__}: {e}")
    return {"mib": None, "method": None, "errors": errors}


def check_floor(floor_mib: int, label: str, python: str | None = None) -> dict:
    """Refuse on a measured value below the floor; warn and proceed when nothing could measure. Returns the probe."""
    probe = free_mib(python)
    if probe["mib"] is None:
        print(f"WARNING: free memory unknown ({'; '.join(probe['errors'])}); proceeding", file=sys.stderr, flush=True)
        probe["warning"] = "free memory unknown; proceeded"
    elif probe["mib"] < floor_mib:
        raise SystemExit(f"REFUSING: {probe['mib']} MiB free on the card ({probe['method']}), {label} {floor_mib}")
    return probe
