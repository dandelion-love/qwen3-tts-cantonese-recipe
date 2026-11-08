# SPDX-License-Identifier: Apache-2.0
"""Resume a run as the SAME DRAW: the data it had not yet seen, in the order it would have seen it, with the RNG and
(for LoRA) the optimizer where they were.

Before this, `--resume_from` restored the weights, the step count and the learning-rate position, and then replayed
the data from epoch 0, batch 0, with AdamW's moments reset. A run resumed at step N trained its first rows twice, ran
the learning rate to zero before the data ended, and in a phase that bakes a voice, put the replayed clips into the
voice's mean twice. So a resumed run was a different draw, and could only count as an attempt.

What a checkpoint now records, and what a resume does with it:

* **The data position**: the epoch and the micro-batches consumed in it. The resume fast-forwards every earlier
  epoch by starting its iterator and taking one batch, which makes exactly the draws an uninterrupted run makes on
  the seeded generator (the loader's base seed, then the sampler's permutation). It then skips the consumed
  micro-batches of the current epoch WITHOUT a forward or backward pass. Skipping by `continue` rather than with
  `accelerate.skip_first_batches` is deliberate: a new loader would draw its seeds in another order, and the
  permutation would change.
* **The generator's state at the checkpoint**, so the resume can PROVE the order matches: at the first real batch
  the generator must be in the state it was in when the checkpoint was written. If it is not, the resume refuses
  rather than training a different draw and calling it the same one.
* **The RNG states** (Python, NumPy, torch and CUDA), so dropout and every other draw continue where they were.
* **The optimizer state, for LoRA.** A full fine-tune's AdamW moments and fp32 masters are about 19.5 GiB at talker
  scope and are still not saved; that choice is recorded in the checkpoint, and such a resume is not the same draw.

Without a data seed (`--seed`) the order cannot be reproduced. The resume then skips by count, says so, and records
that the run is not the same draw.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

RNG_FILE = "rng_state.pt"
OPTIMIZER_FILE = "optimizer.pt"
DATA_GEN_FILE = "data_generator_state.pt"


def capture_rng() -> dict:
    """Every RNG a training step may draw from."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save(directory: str, *, epoch: int, micro_batches: int, optimizer=None, data_generator=None) -> dict:
    """Write what an exact resume needs into `directory` (inside the checkpoint's atomic rename), and return the
    fields for resume_state.json. `optimizer` is None when its state is deliberately not saved (full fine-tune)."""
    torch.save(capture_rng(), os.path.join(directory, RNG_FILE))
    fields = {"epoch": epoch, "epoch_micro_batches": micro_batches,
              "optimizer_state": optimizer is not None,
              "data_order": "seeded" if data_generator is not None else "unseeded"}
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(directory, OPTIMIZER_FILE))
    if data_generator is not None:
        torch.save(data_generator.get_state(), os.path.join(directory, DATA_GEN_FILE))
    return fields


@dataclass
class Position:
    """Where a resumed run starts, and what it can and cannot reproduce."""
    epoch: int = 0
    skip: int = 0
    exact_data: bool = False
    exact_optimizer: bool = False
    generator_state: torch.Tensor | None = None
    rng: dict | None = None

    @property
    def same_draw(self) -> bool:
        return self.exact_data and self.exact_optimizer

    def describe(self) -> str:
        parts = [f"epoch {self.epoch}, skipping {self.skip} consumed micro-batches",
                 "data order: " + ("seeded, checked at the first batch" if self.exact_data
                                   else "UNSEEDED, so the skipped batches are not the consumed ones"),
                 "optimizer: " + ("restored" if self.exact_optimizer
                                  else "NOT restored (full fine-tune, by design, or an old checkpoint)")]
        return "; ".join(parts) + (" -- the SAME draw" if self.same_draw else " -- NOT the same draw: an attempt")


def load(directory: str, state: dict, *, optimizer=None, data_generator=None) -> Position:
    """Restore the optimizer when saved and wanted; return the data position and the RNG state to restore.

    The RNG is NOT restored here: the caller fast-forwards the loader first (a collate may draw), then calls
    `restore_rng(pos.rng)`. A checkpoint written before this module has no epoch_micro_batches: it resumes as before
    (epoch 0, no skip) and reports that it is not the same draw."""
    if "epoch_micro_batches" not in state:
        return Position()
    pos = Position(epoch=int(state["epoch"]), skip=int(state["epoch_micro_batches"]))
    opt_path = os.path.join(directory, OPTIMIZER_FILE)
    if optimizer is not None and os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
        pos.exact_optimizer = True
    gen_path = os.path.join(directory, DATA_GEN_FILE)
    if data_generator is not None and os.path.isfile(gen_path):
        pos.generator_state = torch.load(gen_path, weights_only=False)
        pos.exact_data = True
    pos.rng = torch.load(os.path.join(directory, RNG_FILE), weights_only=False)
    return pos


def fast_forward(dataloader, epochs: int) -> None:
    """Make the data generator's draws for `epochs` whole epochs without loading or training on any batch.

    An epoch draws on the loader's seeded generator three times (torch 2.4): the iterator's base seed when it starts
    (`_BaseDataLoaderIter.__init__`), the sampler's permutation at the first batch, and one more permutation when
    the sampler is exhausted at the end of the epoch (`RandomSampler.__iter__` draws the remainder even when it is
    empty). Taking one batch per epoch misses the third draw. So this replays the base-seed draw and walks the batch
    sampler's indices to exhaustion. It leans on torch's internals; `check_order` proves the result on every resume,
    and refuses if a torch or accelerate version draws differently.

    Restore the RNG states AFTER this.
    """
    gen = getattr(dataloader, "generator", None)
    batch_sampler = dataloader.batch_sampler
    for _ in range(epochs):
        if gen is not None:
            torch.empty((), dtype=torch.int64).random_(generator=gen)      # the iterator's base seed
        for _ in batch_sampler:                                            # the permutation, then the remainder
            pass


def check_order(pos: Position, data_generator) -> None:
    """At the first batch the resumed run trains on, the data generator must be where it was at the checkpoint."""
    if pos.generator_state is None or data_generator is None:
        return
    if not torch.equal(data_generator.get_state(), pos.generator_state):
        raise SystemExit(
            "RESUME refused: the data generator is not in the state it was in at the checkpoint, so the batches "
            "from here on would not be the ones an uninterrupted run trains on. Something between the loader and "
            "the sampler draws differently from the run that wrote the checkpoint.")
