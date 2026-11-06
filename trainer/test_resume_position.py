# SPDX-License-Identifier: Apache-2.0
"""A resumed run is the same draw: the rows it trains on and every loss after the resume equal an uninterrupted run's,
bit for bit on CPU, and each thing the resume restores is shown to matter by a plant that leaves it out.

    python trainer/test_resume_position.py

Needs torch; runs through accelerate when it is installed (as the trainer does), else through a plain loop.

What it is: a replica of the trainer's loop shape (a seeded shuffled DataLoader, gradient accumulation, AdamW, a
linear schedule, dropout, several epochs, a checkpoint at an optimizer step inside an epoch) driving a tiny model,
killed at step 7 (epoch 1, 4 micro-batches in) and resumed from fresh objects with `resume_position`.

What it is not: the trainer. The trainer's own use of these calls is checked by reading it (`test_trainer_wiring`),
and a resume of the real model on a GPU, compared with an uninterrupted run, is a separate measurement.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_position as rp  # noqa: E402

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - the plain path is the fallback
    Accelerator = None
if os.environ.get("RESUME_TEST_PLAIN"):                     # force the plain loop, for comparison
    Accelerator = None

ROWS, BATCH, ACCUM, EPOCHS, KILL_AT, SEED = 20, 2, 2, 3, 7, 1234


class Rows(Dataset):
    def __len__(self):
        return ROWS

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"id": i, "x": torch.randn(8, generator=g), "y": torch.randn(1, generator=g)}


def collate(items):
    return {"id": [it["id"] for it in items], "x": torch.stack([it["x"] for it in items]),
            "y": torch.stack([it["y"] for it in items])}


def run(ckpt_dir, *, kill_at=None, resume=False, plant=None):
    """Train and return (row ids per micro-batch trained, loss per optimizer step after any resume)."""
    torch.manual_seed(SEED if not resume else 999)          # a resumed run starts from a DIFFERENT init
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Dropout(0.3), torch.nn.Linear(16, 1))
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(Rows(), batch_size=BATCH, shuffle=True, collate_fn=collate, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    total = EPOCHS * (ROWS // BATCH // ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))
    acc = Accelerator(gradient_accumulation_steps=ACCUM, cpu=True) if Accelerator else None
    if acc:
        model, opt, loader = acc.prepare(model, opt, loader)
    steps, pos = 0, rp.Position()
    if resume:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), weights_only=False))
        state = torch.load(os.path.join(ckpt_dir, "state.pt"), weights_only=False)
        pos = rp.load(ckpt_dir, state, optimizer=None if plant == "no-optimizer" else opt,
                      data_generator=gen)
        steps = state["optimizer_steps"]
        for _ in range(steps):
            sched.step()
        if plant != "no-fast-forward":
            rp.fast_forward(loader, pos.epoch)
        if plant != "no-rng":
            rp.restore_rng(pos.rng)
        if plant == "no-skip":
            pos.skip = 0
    seen, losses, checked = [], [], False
    for epoch in range(pos.epoch, EPOCHS):
        micro = 0
        for step, batch in enumerate(loader):
            if epoch == pos.epoch and step < pos.skip:
                continue                                    # consumed before the checkpoint: no forward, no backward
            if resume and not checked and plant is None:
                rp.check_order(pos, gen)
                checked = True
            seen.append(tuple(batch["id"]))
            micro += 1
            ctx = acc.accumulate(model) if acc else _null()
            with ctx:
                loss = torch.nn.functional.mse_loss(model(batch["x"]), batch["y"])
                (acc.backward(loss) if acc else (loss / ACCUM).backward())
                sync = acc.sync_gradients if acc else (step + 1) % ACCUM == 0
                if sync:
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    steps += 1
                    losses.append(float(loss.detach()))
                    if kill_at is not None and steps == kill_at:
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                        fields = rp.save(ckpt_dir, epoch=epoch, micro_batches=step + 1, optimizer=opt,
                                         data_generator=gen)
                        torch.save({"optimizer_steps": steps, **fields}, os.path.join(ckpt_dir, "state.pt"))
                        return seen, losses
    return seen, losses


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ResumeIsTheSameDraw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as td:
            cls.full_seen, cls.full_losses = run(td)
        cls.plain = Accelerator is None

    def resumed(self, plant=None):
        with tempfile.TemporaryDirectory() as td:
            seen1, losses1 = run(td, kill_at=KILL_AT)
            seen2, losses2 = run(td, resume=True, plant=plant)
        return seen1 + seen2, losses1 + losses2

    def test_uninterrupted_and_resumed_are_identical(self):
        seen, losses = self.resumed()
        self.assertEqual(seen, self.full_seen, "the resumed run trained on other rows")
        self.assertEqual(losses, self.full_losses, "the losses after the resume differ")   # bit for bit
        self.assertGreater(len(losses), KILL_AT + 3, "the resume must train past the kill point")

    def test_each_restored_piece_is_needed(self):
        for plant in ("no-fast-forward", "no-skip", "no-optimizer", "no-rng"):
            with self.subTest(plant=plant):
                seen, losses = self.resumed(plant)
                self.assertTrue(seen != self.full_seen or losses != self.full_losses,
                                f"{plant}: the resume still matched, so the check cannot see that piece")

    def test_a_wrong_order_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            run(td, kill_at=KILL_AT)
            state = torch.load(os.path.join(td, "state.pt"), weights_only=False)
            gen = torch.Generator().manual_seed(SEED)
            pos = rp.load(td, state, data_generator=gen)
            with self.assertRaises(SystemExit):                 # not fast-forwarded: the generator is elsewhere
                rp.check_order(pos, gen)

    def test_an_old_checkpoint_resumes_as_before(self):
        pos = rp.load("/nonexistent", {"optimizer_steps": 5, "epoch": 1})
        self.assertEqual((pos.epoch, pos.skip, pos.same_draw), (0, 0, False))
        self.assertIn("NOT the same draw", pos.describe())


class TrainerWiring(unittest.TestCase):
    """The trainer calls these in the order this test runs them. Read from its source, not run."""

    def test_trainer_wiring(self):
        src = (Path(__file__).resolve().parent / "sft_12hz_lora_mlflow.py").read_text(encoding="utf-8")
        order = ["rp.load(", "rp.fast_forward(", "rp.restore_rng(", "range(_resume.epoch, args.num_epochs)",
                 "step < _resume.skip", "rp.check_order("]
        at = [src.find(s) for s in order]
        self.assertTrue(all(i >= 0 for i in at), dict(zip(order, at)))
        self.assertEqual(at, sorted(at), "the resume calls are out of order in the trainer")
        self.assertIn("rp.save(", src)
        self.assertIn("micro_batches=step + 1", src)


if __name__ == "__main__":
    print("accelerate:", "yes" if Accelerator else "no (plain loop)")
    unittest.main(verbosity=2)
