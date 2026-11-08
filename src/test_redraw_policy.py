"""CPU-only tests for demo-only redraws and retained-draw records."""
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import render_story as story


class RedrawTests(unittest.TestCase):
    def test_cli_refuses_redraw_before_writing_or_loading(self):
        for value in ("1", "-1"):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "out"
                argv = ["render_story", "--texts", d, "--out", str(out), "--arm", "a=model:voice", "--redraw", value]
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exc:
                        story.main()
                self.assertEqual(exc.exception.code, 2)
                self.assertFalse(out.exists())

    def test_refusal_plant_and_allowed_paths(self):
        self.assertIsNotNone(story.redraw_refusal(2, False))
        self.assertIsNone(story.redraw_refusal(2, True))
        self.assertIsNone(story.redraw_refusal(0, False))
        self.assertIsNotNone(story.redraw_refusal(-1, True))
        # If the check were replaced with an unconditional allow, the non-demo case fails.
        def check(f):
            return f(2, False) is not None and f(2, True) is None
        self.assertTrue(check(story.redraw_refusal))
        self.assertFalse(check(lambda *_: None))

    def test_cli_guard_mutant_is_detected(self):
        # Replace the real guard with an unconditional allow: the same CLI request
        # that must refuse now reaches a manifest, so the CLI rejection test turns red.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "out"
            audio = types.ModuleType("audio_uri")
            audio.write_flac = lambda *_: None
            argv = ["render_story", "--texts", d, "--out", str(out), "--arm", "a=model:voice", "--redraw", "1"]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"audio_uri": audio}), \
                 mock.patch.object(story, "redraw_refusal", return_value=None), \
                 mock.patch.object(story, "engine_block", return_value={}), \
                 mock.patch.object(story, "require_provenance"), \
                 mock.patch.object(story, "render_arm", return_value={}):
                self.assertEqual(story.main(), 0)
            self.assertTrue((out / "manifest.json").exists())

    def test_actual_loop_retains_first_or_last_and_records_why(self):
        chunk = {"text": "你好。", "punct": "。", "last_in_para": True}
        cases = [
            (0, [("silent", 0.01)], "first draw", 0, "silent"),
            (2, [("silent", 0.01), (None, 0.20)], "a clean redraw", 1, None),
            (2, [("silent", 0.01), ("runaway", 0.90), ("silent", 0.02)],
             "the last draw: every redraw was defective", 2, "silent"),
        ]
        for redraw, judgements, kept, index, final in cases:
            with self.subTest(kept=kept):
                calls = []
                class FakeModel:
                    def __init__(self, *_a, **_kw): pass
                    def generate_custom_voice(self, **kw):
                        calls.append(kw["seed"])
                        # Distinct sample patterns survive the render's level scaling.
                        y = np.where(np.arange(100) % (len(calls) + 1) == 0, -0.1, 0.1)
                        return [y.astype("float32")], 1000
                    def close(self): pass
                gb = types.ModuleType("gguf_backend")
                gb.GGUFModel = FakeModel
                trim = types.ModuleType("audio_trim")
                trim.trim_edges = lambda y, sr, **kw: (y, {})
                with mock.patch.dict(sys.modules, {"gguf_backend": gb, "audio_trim": trim}), \
                     mock.patch.object(story, "judge", side_effect=judgements), contextlib.redirect_stdout(io.StringIO()):
                    result = story.render_arm("a", Path("model"), "voice", {"s": ("title", [chunk])}, Path("."), redraw)
                row = result["s"]["chunks"][0]
                self.assertEqual((row["redraws"], row["kept"], row["final_defect"]), (index, kept, final))
                self.assertEqual(row["first_draw_defect"], "silent")
                self.assertEqual(row["first_s_per_char"], 0.01)
                self.assertEqual(len(calls), index + 1)
                expected = np.where(np.arange(100) % (index + 2) == 0, -1, 1)
                np.testing.assert_array_equal(np.sign(result["s"]["read"]), expected)
                self.assertEqual(row["seed"], story.seed_of("s", 0, chunk["text"], index))

    def test_demo_declaration_is_written_to_manifest(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            texts, out = root / "texts", root / "out"
            texts.mkdir()
            (texts / "s.txt").write_text("title\nA sample sentence.\n", encoding="utf8")
            fake_audio = types.ModuleType("audio_uri")
            fake_audio.write_flac = lambda path, *_: path.write_bytes(b"synthetic")
            fake_story = {"title": "title", "chunks": [{"kept": "a clean redraw"}],
                          "read": np.array([0.1], dtype="float32"), "sr": 1000}
            argv = ["render_story", "--texts", str(texts), "--out", str(out), "--arm", "a=model:voice",
                    "--redraw", "1", "--published-demo"]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"audio_uri": fake_audio}), \
                 mock.patch.object(story, "engine_block", return_value={}), \
                 mock.patch.object(story, "require_provenance"), \
                 mock.patch.object(story, "render_arm", return_value={"s": fake_story}):
                self.assertEqual(story.main(), 0)
            man = json.loads((out / "manifest.json").read_text(encoding="utf8"))
            self.assertTrue(man["rule"]["published_demo"])
            self.assertEqual(man["rule"]["max_redraw"], 1)
            self.assertEqual(man["arms"]["a"]["stories"]["s"]["chunks"][0]["kept"], "a clean redraw")

    def test_checkpoint_card_refuses_redraw(self):
        import checkpoint_card as card
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text(json.dumps({"rule": {"max_redraw": 1}}), encoding="utf8")
            steps = types.ModuleType("steps")
            fake = {"steps": steps, "skipped_text": types.ModuleType("skipped_text")}
            with mock.patch.dict(sys.modules, fake), mock.patch.object(card, "_ASR", {"q": object(), "s": object()}), \
                 mock.patch.object(story, "require_provenance"):
                with self.assertRaisesRegex(SystemExit, "not first draws"):
                    card.dropped(root, "a")


if __name__ == "__main__":
    unittest.main()
