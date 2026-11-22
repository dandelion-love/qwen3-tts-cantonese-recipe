# SPDX-License-Identifier: Apache-2.0
"""Full fine-tuning's 8-bit AdamW and frozen text embedding, on a tiny model, through the trainer's own code.

    python trainer/test_full_ft_optim.py                          # CPU
    FULLFT_TEST_DEVICE=cuda python trainer/test_full_ft_optim.py  # the same 20-step runs on the GPU (smoke)

Needs the trainer's imports (torch, accelerate, qwen-tts, mlflow, peft). The 8-bit cases need bitsandbytes
(`pip install -r requirements-8bit.txt`) and skip without it; the refusal case runs either way and hides
bitsandbytes itself. bitsandbytes 0.50.2 has a CPU kernel for the 8-bit update, so nothing here needs a GPU.

What it checks:
- 8-bit AdamW trains: 20 steps change the weights on every optimizer configuration the trainer can build with it
  (bf16 weights + fp32 masters, the same with the masters offloaded, fp32 weights, plain bf16), the moments are
  stored as uint8 codes, and the masters are fp32. A zero learning rate leaves every weight bit-identical, so
  "changed" is not an artefact of the harness. Its update tracks fp32 AdamW's (cosine > 0.9).
- The frozen text embedding stays bit-identical through 20 steps and through the bf16 checkpoint cast while every
  other tensor moves, holds no optimizer state and no master; without the freeze the same run moves it (the plant).
- A resume re-seeds the fp32 masters from the resumed weights, so the next step continues from them (without the
  re-seed, the plant, the step writes the pre-resume masters back), for both optimizers, offloaded or not.
- A missing bitsandbytes refuses, from the factory and from the command line, and never falls back to fp32 AdamW.

What it is not: the 0.6B model. Its memory and step time on a 12 GB card are a separate GPU measurement.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_12hz_lora_mlflow as T  # noqa: E402

try:
    import bitsandbytes  # noqa: F401
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False

DEVICE = torch.device(os.environ.get("FULLFT_TEST_DEVICE", "cpu"))
STEPS = 20


class _Talker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.text_embedding = nn.Embedding(64, 64)      # 4096 values: bitsandbytes' 8-bit floor
        self.text_projection = nn.Linear(64, 128)
        self.code_predictor = nn.Linear(128, 128)
        self.codec_head = nn.Linear(128, 16)                  # 2048 values: stays fp32 state


class Tiny(nn.Module):
    """Parameter names as the real model's, so apply_full_ft_scope and freeze_text_embedding see the real layout."""

    def __init__(self):
        super().__init__()
        self.talker = _Talker()

    def forward(self, ids):
        t = self.talker
        h = t.text_projection(t.model.text_embedding(ids))
        return t.codec_head(torch.tanh(t.code_predictor(h)))


def _model(dtype):
    torch.manual_seed(0)
    m = Tiny()
    for p in m.parameters():
        nn.init.normal_(p, std=0.02)
    return m.to(DEVICE, dtype)


def _groups(model):
    """The trainer's grouping: requires_grad parameters, code predictor in its own group."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    cp = [p for n, p in trainable if "code_predictor" in n]
    rest = [p for n, p in trainable if "code_predictor" not in n]
    return [{"params": rest, "lr": 0.0}, {"params": cp, "lr": 0.0}]


def _train(model, opt, steps=STEPS):
    g = torch.Generator().manual_seed(1)
    ids = torch.arange(64).reshape(8, 8).to(DEVICE)          # every embedding row receives gradient
    target = torch.randn(8, 8, 16, generator=g).to(DEVICE)
    for _ in range(steps):
        out = model(ids).float()
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item()


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _run(dtype, *, lr, eightbit, masters, offload=False, freeze=False):
    model = _model(dtype)
    T.apply_full_ft_scope(model, "all")
    if freeze:
        T.freeze_text_embedding(model)
    groups = _groups(model)
    for gr in groups:
        gr["lr"] = lr
    before = _snapshot(model)
    opt = T.make_full_ft_optimizer(groups, lr=lr, weight_decay=0.01 if lr else 0.0,
                                   eightbit=eightbit, masters=masters, offload_master=offload)
    _train(model, opt)
    return model, opt, before


def _moved(model, before):
    return {n: (p.detach() != before[n]).float().mean().item() for n, p in model.named_parameters()}


# (name, weight dtype, masters, offload) -- every way the trainer builds an optimizer under --full_ft
CONFIGS = [
    ("bf16 weights + fp32 masters", torch.bfloat16, True, False),
    ("bf16 weights + fp32 masters offloaded", torch.bfloat16, True, True),
    ("fp32 weights", torch.float32, False, False),
    ("plain bf16", torch.bfloat16, False, False),
]


@unittest.skipUnless(HAVE_BNB, "bitsandbytes not installed (pip install -r requirements-8bit.txt)")
class EightBitTrains(unittest.TestCase):
    """AC1: --full_ft_optimizer adamw8bit trains and changes weights over 20 steps."""

    def test_changes_weights_in_every_configuration(self):
        for name, dtype, masters, offload in CONFIGS:
            with self.subTest(name):
                model, opt, before = _run(dtype, lr=1e-3, eightbit=True, masters=masters, offload=offload)
                moved = _moved(model, before)
                for n, frac in moved.items():
                    self.assertGreater(frac, 0.5, f"{name}: {n} moved {frac:.1%}")
                inner = opt
                self.assertIsInstance(inner, T.require_bitsandbytes().optim.AdamW8bit)
                big = [p for gr in inner.param_groups for p in gr["params"] if p.numel() >= 4096]
                self.assertTrue(big)
                for p in big:
                    st = inner.state[p]
                    self.assertEqual(st["state1"].dtype, torch.uint8, name)
                    self.assertEqual(st["state2"].dtype, torch.uint8, name)
                if masters:
                    devs = {m.device.type for _, m in opt._pairs}
                    self.assertEqual({m.dtype for _, m in opt._pairs}, {torch.float32})
                    self.assertEqual(devs, {"cpu"} if offload else {DEVICE.type})

    def test_zero_lr_is_a_no_op(self):
        # The control for the test above: the harness alone must not change a weight.
        for name, dtype, masters, offload in CONFIGS:
            with self.subTest(name):
                model, _, before = _run(dtype, lr=0.0, eightbit=True, masters=masters, offload=offload)
                for n, p in model.named_parameters():
                    self.assertTrue(torch.equal(p.detach(), before[n]), f"{name}: {n} changed at lr 0")

    def test_tracks_fp32_adamw(self):
        m8, _, before = _run(torch.float32, lr=1e-3, eightbit=True, masters=False)
        m32, _, _ = _run(torch.float32, lr=1e-3, eightbit=False, masters=False)
        d8 = torch.cat([(p.detach() - before[n]).flatten() for n, p in m8.named_parameters()])
        d32 = torch.cat([(p.detach() - before[n]).flatten() for n, p in m32.named_parameters()])
        cos = torch.nn.functional.cosine_similarity(d8.float(), d32.float(), dim=0).item()
        self.assertGreater(cos, 0.9, cos)


class FrozenTextEmbedding(unittest.TestCase):
    """AC2: --freeze_text_embedding under --full_ft leaves that tensor bit-identical while others move."""

    def _optimizers(self):
        out = [("fp32 AdamW + masters", False)]
        if HAVE_BNB:
            out.append(("8-bit AdamW + masters", True))
        return out

    def test_frozen_tensor_is_bit_identical_and_others_move(self):
        for name, eightbit in self._optimizers():
            with self.subTest(name):
                model, opt, before = _run(torch.bfloat16, lr=1e-3, eightbit=eightbit, masters=True, freeze=True)
                emb = dict(model.named_parameters())[T.TEXT_EMBEDDING_PARAM]
                self.assertTrue(torch.equal(emb.detach(), before[T.TEXT_EMBEDDING_PARAM]))
                self.assertIsNone(emb.grad)
                self.assertNotIn(id(emb), {id(p) for p, _ in opt._pairs}, "frozen table has a master")
                self.assertEqual(len(opt._pairs), sum(1 for _ in model.parameters()) - 1)
                for n, frac in _moved(model, before).items():
                    if n != T.TEXT_EMBEDDING_PARAM:
                        self.assertGreater(frac, 0.5, f"{name}: {n} moved {frac:.1%}")
                # and the checkpoint carries it unchanged
                sd = T.full_ft_state_dict(model)
                self.assertTrue(torch.equal(sd[T.TEXT_EMBEDDING_PARAM],
                                            before[T.TEXT_EMBEDDING_PARAM].cpu().to(torch.bfloat16)))

    def test_plant_without_freeze_it_moves(self):
        model, _, before = _run(torch.bfloat16, lr=1e-3, eightbit=False, masters=True, freeze=False)
        self.assertGreater(_moved(model, before)[T.TEXT_EMBEDDING_PARAM], 0.5)

    def test_reports_count_and_scope_no_op(self):
        m = _model(torch.float32)
        T.apply_full_ft_scope(m, "all")
        self.assertEqual(T.freeze_text_embedding(m), 64 * 64)
        m = _model(torch.float32)
        T.apply_full_ft_scope(m, "code_predictor")
        self.assertEqual(T.freeze_text_embedding(m), 0)

    def test_refuses_shared_storage(self):
        m = _model(torch.float32)
        m.talker.tied = m.talker.model.text_embedding          # a tie would freeze the other user too
        m.talker.tied_head = nn.Linear(64, 64, bias=False)
        m.talker.tied_head.weight = m.talker.model.text_embedding.weight
        with self.assertRaisesRegex(RuntimeError, "shares storage"):
            T.freeze_text_embedding(m)

    def test_refuses_missing_table(self):
        with self.assertRaisesRegex(RuntimeError, "not found"):
            T.freeze_text_embedding(nn.Linear(2, 2))


class MissingBitsandbytesRefuses(unittest.TestCase):
    """AC3: bitsandbytes missing gives a clear refusal, not a fallback."""

    def _hidden(self):
        return mock.patch.dict(sys.modules, {"bitsandbytes": None})   # import raises ImportError

    def test_factory_refuses_with_and_without_masters(self):
        with self._hidden():
            for masters in (True, False):
                with self.subTest(masters=masters):
                    p = nn.Parameter(torch.zeros(4))
                    with self.assertRaises(T.MissingBitsandbytes) as cm:
                        T.make_full_ft_optimizer([{"params": [p]}], lr=1e-3, eightbit=True, masters=masters)
                    msg = str(cm.exception)
                    self.assertIn("requirements-8bit.txt", msg)
                    self.assertIn(T.BITSANDBYTES_PIN, msg)
                    self.assertIn("no fallback", msg)

    def _cli(self, argv):
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["sft_12hz_lora_mlflow.py", *argv]), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            T.train()
        return cm.exception.code, err.getvalue()

    def test_command_line_refuses_before_loading(self):
        with self._hidden():
            code, err = self._cli(["--full_ft", "--full_ft_optimizer", "adamw8bit",
                                   "--train_jsonl", "unused.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("needs bitsandbytes", err)
        self.assertIn("no fallback", err)

    def test_options_need_full_ft(self):
        code, err = self._cli(["--full_ft_optimizer", "adamw8bit", "--train_jsonl", "unused.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("--full_ft only", err)
        code, err = self._cli(["--freeze_text_embedding", "--train_jsonl", "unused.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("--full_ft only", err)

    def test_offload_needs_masters(self):
        code, err = self._cli(["--full_ft", "--offload_master", "--train_jsonl", "unused.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("bf16_master_fp32", err)


class ResumeReseedsMasters(unittest.TestCase):
    """The trainer's resume order: optimizer built (masters cloned), then resumed weights loaded, then a step.

    With the re-seed the step continues from the resumed weights; without it (the plant) the step writes the
    pre-resume masters back and the resume is undone.
    """

    NAME = "talker.text_projection.weight"

    def _resume(self, *, eightbit, offload, reseed):
        model = _model(torch.bfloat16)
        T.apply_full_ft_scope(model, "all")
        groups = _groups(model)
        for gr in groups:
            gr["lr"] = 1e-6
        opt = T.make_full_ft_optimizer(groups, lr=1e-6, weight_decay=0.0, eightbit=eightbit,
                                       masters=True, offload_master=offload)
        resumed = {k: (v + 0.5).to(v.dtype) for k, v in model.state_dict().items()}
        model.load_state_dict(resumed)
        n = T.reseed_masters(opt) if reseed else 0
        _train(model, opt, steps=1)
        w = dict(model.named_parameters())[self.NAME].detach().float()
        return n, (w - resumed[self.NAME].float()).abs().max().item(), len(opt._pairs)

    def _configs(self):
        out = [("adamw", False, False), ("adamw offloaded", False, True)]
        if HAVE_BNB:
            out += [("adamw8bit", True, False), ("adamw8bit offloaded", True, True)]
        return out

    def test_step_continues_from_resumed_weights(self):
        for name, eightbit, offload in self._configs():
            with self.subTest(name):
                n, drift, pairs = self._resume(eightbit=eightbit, offload=offload, reseed=True)
                self.assertEqual(n, pairs)
                self.assertLess(drift, 0.01, f"{name}: drifted {drift} from the resumed weights")

    def test_plant_without_reseed_the_resume_is_undone(self):
        for name, eightbit, offload in self._configs():
            with self.subTest(name):
                _, drift, _ = self._resume(eightbit=eightbit, offload=offload, reseed=False)
                self.assertGreater(drift, 0.4, f"{name}: the plant did not show the hazard")

    def test_no_masters_is_a_no_op(self):
        p = nn.Parameter(torch.zeros(4))
        self.assertEqual(T.reseed_masters(torch.optim.AdamW([p])), 0)

    def test_not_restored_names_the_state(self):
        p = nn.Parameter(torch.zeros(8))
        self.assertIn("exp_avg", T.full_ft_not_restored(torch.optim.AdamW([p])))
        self.assertIn("masters", T.full_ft_not_restored(
            T.make_full_ft_optimizer([{"params": [p]}], lr=1e-3, masters=True)))
        if HAVE_BNB:
            msg = T.full_ft_not_restored(T.make_full_ft_optimizer([{"params": [p]}], lr=1e-3,
                                                                  eightbit=True, masters=True))
            self.assertIn("state1/state2", msg)
            self.assertIn("masters", msg)


class TrainerWiring(unittest.TestCase):
    """The trainer builds its optimizer through make_full_ft_optimizer, and freezes before it groups parameters."""

    def test_wiring(self):
        src = (HERE / "sft_12hz_lora_mlflow.py").read_text(encoding="utf-8")
        order = ["apply_full_ft_scope(train_model", "freeze_text_embedding(train_model)",
                 "trainable = [(n, p) for n, p in train_model.named_parameters() if p.requires_grad]",
                 "optimizer = make_full_ft_optimizer("]
        at = [src.find(s) for s in order]
        self.assertTrue(all(i >= 0 for i in at), dict(zip(order, at)))
        self.assertEqual(at, sorted(at))
        self.assertEqual(src.count("optimizer = make_full_ft_optimizer("), 2)   # masters and plain branches
        self.assertNotIn("optimizer = AdamW(", src)
        self.assertNotIn("optimizer = MasterWeightAdamW(", src)

    def test_resume_reseeds_after_loading(self):
        src = (HERE / "sft_12hz_lora_mlflow.py").read_text(encoding="utf-8")
        start = src.find("if args.resume_from:\n            import safetensors.torch")
        order = ["load_state_dict(", "reseed_masters(optimizer)", "for _ in range(optimizer_steps):"]
        at = [src.find(s, start) for s in order]
        self.assertTrue(start >= 0 and all(i > start for i in at), dict(zip(order, at)))
        self.assertEqual(at, sorted(at))
        self.assertIn("not_restored=(full_ft_not_restored(optimizer)", src)


if __name__ == "__main__":
    print(f"device: {DEVICE}  bitsandbytes: {getattr(sys.modules.get('bitsandbytes'), '__version__', 'missing')}")
    unittest.main(verbosity=2)
