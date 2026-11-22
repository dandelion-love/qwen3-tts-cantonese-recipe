# SPDX-License-Identifier: Apache-2.0
"""--full_ft_precision bf16_sr: bf16 weights, 8-bit AdamW, stochastic rounding, no fp32 masters. CPU.

    python trainer/test_full_ft_sr.py

Needs the trainer's imports and bitsandbytes (`pip install -r requirements-8bit.txt`); the cases that use the
optimizer skip without it. bitsandbytes 0.50.2 has a CPU kernel for the 8-bit update, so nothing here needs a GPU.

What it checks:
- The rounding is unbiased: the mean of many stochastic roundings of a value between two bf16 neighbours is that
  value, for positive and negative values, while round-to-nearest is off by the dropped fraction.
- At lr 2e-6, where round-to-nearest bf16 keeps almost every weight unchanged, stochastic rounding moves them, and
  its change points the same way as the fp32-master run's.
- At lr 0 every weight is bit-identical, so "moved" is not an artefact of the rounding.
- The trainer refuses bf16_sr without 8-bit AdamW, and the factory refuses it with masters.
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_12hz_lora_mlflow as T  # noqa: E402
from test_full_ft_optim import _groups, _model, _moved, _snapshot, _train  # noqa: E402

try:
    import bitsandbytes  # noqa: F401
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False

ULP_AT_ONE = 2.0 ** -7          # bf16 spacing in [1, 2)


class RoundingIsUnbiased(unittest.TestCase):
    def test_mean_is_the_value(self):
        torch.manual_seed(0)
        n = 200_000
        for frac in (0.1, 0.3, 0.5, 0.9):
            for sign in (1.0, -1.0):
                with self.subTest(frac=frac, sign=sign):
                    x = torch.full((n,), sign * (1.0 + frac * ULP_AT_ONE), dtype=torch.float32)
                    r = T.stochastic_round_bf16(x)
                    self.assertEqual(r.dtype, torch.bfloat16)
                    # only the two neighbours ever appear
                    vals = set(r.float().abs().unique().tolist())
                    self.assertTrue(vals <= {1.0, 1.0 + ULP_AT_ONE}, vals)
                    err = (r.float().mean().item() - x[0].item()) / ULP_AT_ONE
                    self.assertLess(abs(err), 0.01, f"bias {err:+.4f} ulp")
                    rtn = (x[0].to(torch.bfloat16).float().item() - x[0].item()) / ULP_AT_ONE
                    if frac != 0.5:
                        self.assertGreater(abs(rtn), 0.09, "round-to-nearest should be biased here")

    def test_exact_and_non_finite_values_pass_through(self):
        x = torch.tensor([0.0, -0.0, 1.0, -2.5, 2.0 ** 100, float("inf"), float("-inf")])
        self.assertTrue(torch.equal(T.stochastic_round_bf16(x).float(), x.to(torch.bfloat16).float()))
        self.assertTrue(T.stochastic_round_bf16(torch.tensor([float("nan")])).isnan().all())


@unittest.skipUnless(HAVE_BNB, "bitsandbytes not installed (pip install -r requirements-8bit.txt)")
class StochasticRoundingTrains(unittest.TestCase):
    STEPS = 50

    def _run(self, *, lr, stochastic_round=False, masters=False):
        torch.manual_seed(0)
        model = _model(torch.bfloat16)
        T.apply_full_ft_scope(model, "all")
        groups = _groups(model)
        for gr in groups:
            gr["lr"] = lr
        before = _snapshot(model)
        opt = T.make_full_ft_optimizer(groups, lr=lr, weight_decay=0.01 if lr else 0.0, eightbit=True,
                                       masters=masters, stochastic_round=stochastic_round)
        _train(model, opt, steps=self.STEPS)
        return model, opt, before

    def _fraction(self, model, before):
        moved = _moved(model, before)
        n = {k: p.numel() for k, p in model.named_parameters()}
        return sum(moved[k] * n[k] for k in n) / sum(n.values())

    def test_moves_where_round_to_nearest_stalls(self):
        sr, opt, sr0 = self._run(lr=2e-6, stochastic_round=True)
        rtn, _, rtn0 = self._run(lr=2e-6)
        f_sr, f_rtn = self._fraction(sr, sr0), self._fraction(rtn, rtn0)
        print(f"\n  lr 2e-6, {self.STEPS} steps: moved {f_sr:.1%} with stochastic rounding, "
              f"{f_rtn:.1%} with round-to-nearest")
        self.assertGreater(f_sr, 0.10)
        self.assertGreater(f_sr, 10 * f_rtn)
        for p in sr.parameters():
            self.assertEqual(p.dtype, torch.bfloat16)
        big = [p for p in sr.parameters() if p.numel() >= 4096]
        self.assertTrue(big and all(opt.state[p]["state1"].dtype == torch.uint8 for p in big))
        # The direction agrees with the fp32-master run of the same steps.
        ref, _, ref0 = self._run(lr=2e-6, masters=True)
        d_sr = torch.cat([(p.float() - sr0[n].float()).flatten() for n, p in sr.named_parameters()])
        d_ref = torch.cat([(p.float() - ref0[n].float()).flatten() for n, p in ref.named_parameters()])
        cos = torch.nn.functional.cosine_similarity(d_sr, d_ref, dim=0).item()
        print(f"  cosine to the fp32-master run's change: {cos:.3f}")
        self.assertGreater(cos, 0.4)

    def test_lr_zero_is_bit_identical(self):
        model, _, before = self._run(lr=0.0, stochastic_round=True)
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p.detach().view(torch.int16), before[n].view(torch.int16)), n)


class Refusals(unittest.TestCase):
    def test_factory(self):
        p = nn.Parameter(torch.zeros(4))
        for kw in ({"eightbit": False}, {"eightbit": True, "masters": True}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                T.make_full_ft_optimizer([{"params": [p]}], lr=1e-3, stochastic_round=True, **kw)

    def test_command_line(self):
        err = io.StringIO()
        argv = ["sft_12hz_lora_mlflow.py", "--full_ft", "--full_ft_precision", "bf16_sr",
                "--train_jsonl", "unused.jsonl"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as cm:
            T.train()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("adamw8bit only", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
