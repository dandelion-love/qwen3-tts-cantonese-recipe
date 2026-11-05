# SPDX-License-Identifier: Apache-2.0
"""The card-memory probe: a failure to measure is not a measurement of low memory.

Runs on CPU with the standard library only; every probe is faked, nothing touches a GPU:

    python src/test_card_memory.py

Covers card_memory.free_mib/check_floor directly (nvidia-smi fails once then succeeds; nvidia-smi always fails and
torch works; everything fails; a measured low value), and recipe.train() itself for the two cases that decide a run:
everything fails -> training proceeds with the warning in its record; measured low -> refused before the trainer runs.

NOT covered: a real nvidia-smi or torch on a real card. The faked probes pin the decision logic, not the parsing of
real output (nvidia_smi_mib keeps the parsing the old probe had).
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import card_memory as cm  # noqa: E402


def flaky(*results):
    """A fake probe returning or raising each of `results` in turn."""
    it = iter(results)

    def probe(*_a, **_k):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r
    return probe


class Probe(unittest.TestCase):
    def setUp(self):
        mock.patch.object(cm.time, "sleep", lambda _s: None).start()
        self.addCleanup(mock.patch.stopall)

    def test_nvidia_smi_fails_once_then_succeeds(self):
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(TimeoutError("30 s"), 20000)), \
             mock.patch.object(cm, "torch_mib", flaky(AssertionError("torch must not be reached"))):
            p = cm.free_mib()
        self.assertEqual((p["mib"], p["method"]), (20000, "nvidia-smi"))
        self.assertEqual(len(p["errors"]), 1)

    def test_nvidia_smi_always_fails_torch_works(self):
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(*[OSError("no nvidia-smi")] * cm.PROBE_TRIES)), \
             mock.patch.object(cm, "torch_mib", flaky(18000)):
            p = cm.free_mib()
        self.assertEqual((p["mib"], p["method"]), (18000, "torch"))
        self.assertEqual(len(p["errors"]), cm.PROBE_TRIES)

    def test_everything_fails_warns_and_proceeds(self):
        err = io.StringIO()
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(*[OSError("no nvidia-smi")] * cm.PROBE_TRIES)), \
             mock.patch.object(cm, "torch_mib", flaky(RuntimeError("no CUDA"))), redirect_stderr(err):
            p = cm.check_floor(8000, "[train].min_free_mib")         # must not raise
        self.assertIsNone(p["mib"])
        self.assertIn("warning", p)                                   # what train() writes into the stage record
        self.assertIn("WARNING: free memory unknown (", err.getvalue())
        self.assertIn("no CUDA", err.getvalue())
        self.assertIn("proceeding", err.getvalue())
        self.assertEqual(len(p["errors"]), cm.PROBE_TRIES + 1)

    def test_measured_low_value_refuses(self):
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(5000)):
            with self.assertRaises(SystemExit) as e:
                cm.check_floor(8000, "[train].min_free_mib")
        self.assertIn("REFUSING: 5000 MiB", str(e.exception))

    def test_measured_value_at_floor_passes(self):
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(8000)):
            self.assertEqual(cm.check_floor(8000, "floor")["mib"], 8000)


class RecipeTrain(unittest.TestCase):
    """recipe.train() with the probe faked: it proceeds on unknown and refuses on a measured low value."""

    class Ran(Exception):
        pass

    def setUp(self):
        import recipe
        self.recipe = recipe
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        work = Path(self.td.name)
        self.data = work / "train.jsonl"
        self.data.write_text("{}\n", encoding="utf-8")
        keys = ("batch_size", "gradient_accumulation_steps", "lr", "num_epochs", "warmup_ratio", "lora_rank",
                "lora_alpha", "checkpoint_every_epochs", "checkpoint_every_steps", "sub_talker_weight",
                "attn_implementation", "seed")
        self.ran = []

        def run(stage, cmd, cwd=None, env=None):
            self.ran.append(stage)
            raise self.Ran(stage)
        self.r = SimpleNamespace(cfg={"train": {**{k: 1 for k in keys}, "min_free_mib": 8000}}, work=work, dry=False,
                                 py=sys.executable, run=run)
        mock.patch.object(cm.time, "sleep", lambda _s: None).start()
        self.addCleanup(mock.patch.stopall)

    def test_everything_fails_training_proceeds(self):
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(*[OSError("no nvidia-smi")] * cm.PROBE_TRIES)), \
             mock.patch.object(cm, "torch_mib", flaky(RuntimeError("no CUDA"))), redirect_stderr(io.StringIO()):
            with self.assertRaises(self.Ran):
                self.recipe.train(self.r, "spk", Path("init"), self.data, [])
        self.assertEqual(self.ran, ["train-spk"])

    def test_measured_low_training_refused(self):
        with mock.patch.object(cm, "nvidia_smi_mib", flaky(5000)):
            with self.assertRaises(SystemExit) as e:
                self.recipe.train(self.r, "spk", Path("init"), self.data, [])
        self.assertIn("REFUSING: 5000 MiB", str(e.exception))
        self.assertEqual(self.ran, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
