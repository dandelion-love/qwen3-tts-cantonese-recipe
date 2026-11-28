"""CPU-only tests for the register filter (require_colloquial): it never applies to language data.

recipe.py refuses any [data] set but [data.speaker] configured with require_colloquial = true, and a language set given
already selected whose selection.json records the filter. The speaker set may still opt in. Both selectors default the
filter off. Each refusal is shown to fire, and the shipped recipe.toml is shown to pass.

    python src/test_register_filter.py
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import recipe  # noqa: E402
import select_common_voice  # noqa: E402
import select_mdc  # noqa: E402


def shipped() -> dict:
    return recipe.tomllib.loads((ROOT / "recipe.toml").read_text(encoding="utf-8"))


def with_flag(which: str, on: bool) -> dict:
    cfg = json.loads(json.dumps(shipped()))
    cfg["data"][which]["require_colloquial"] = on
    return cfg


class RegisterFilterTests(unittest.TestCase):
    def test_shipped_recipe_passes_and_language_sets_are_off(self):
        cfg = shipped()
        recipe.check_register(cfg)
        for which in ("language", "zhhk"):
            self.assertIs(cfg["data"][which]["require_colloquial"], False, which)

    def test_shipped_recipe_loads(self):
        recipe.Recipe(ROOT / "recipe.toml", dry=True)

    def test_refuses_the_yue_language_set(self):
        with self.assertRaises(SystemExit) as e:
            recipe.check_register(with_flag("language", True))
        self.assertIn("[data.language]", str(e.exception))

    def test_refuses_the_zhhk_language_set(self):
        with self.assertRaises(SystemExit) as e:
            recipe.check_register(with_flag("zhhk", True))
        self.assertIn("[data.zhhk]", str(e.exception))

    def test_refuses_any_other_non_speaker_set(self):
        cfg = shipped()
        cfg["data"]["extra"] = dict(cfg["data"]["zhhk"], require_colloquial=True)
        with self.assertRaises(SystemExit) as e:
            recipe.check_register(cfg)
        self.assertIn("[data.extra]", str(e.exception))

    def test_speaker_set_may_opt_in(self):
        recipe.check_register(with_flag("speaker", True))

    def test_recipe_refuses_at_load(self):
        with tempfile.TemporaryDirectory() as t:
            text = (ROOT / "recipe.toml").read_text(encoding="utf-8")
            head, tail = text.split("[data.zhhk]", 1)
            tail = tail.replace("require_colloquial = false", "require_colloquial = true", 1)
            p = Path(t) / "recipe.toml"
            p.write_text(head + "[data.zhhk]" + tail, encoding="utf-8")
            with self.assertRaises(SystemExit):
                recipe.Recipe(p, dry=True)

    def test_dry_select_passes_the_filter_off_to_language_selectors(self):
        r = recipe.Recipe(ROOT / "recipe.toml", dry=True)
        r.paths = {**r.paths, "work": "/nonexistent-work", "cv_root": "/cv", "zhhk_mdc_archive": "/zh.tar.gz",
                   "lang_selected_dir": "", "speaker_selected_dir": "", "donor_ids": "/ids"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            recipe.stage_select(r)
        selects = [ln for ln in out.getvalue().splitlines() if "select_" in ln]
        self.assertEqual(len(selects), 3, out.getvalue())
        for ln in selects:
            self.assertIn("--no-require-colloquial", ln)
            self.assertNotIn(" --require-colloquial", ln)

    def test_preselected_language_set_selected_with_the_filter_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            sel = Path(t)
            (sel / "selection.json").write_text(json.dumps({"filters": {"require_colloquial": True}}),
                                                encoding="utf-8")
            with self.assertRaises(SystemExit):
                recipe.check_preselected_register("language", sel)
            recipe.check_preselected_register("speaker", sel)          # the speaker set may have opted in
            (sel / "selection.json").write_text(json.dumps({"filters": {"require_colloquial": False}}),
                                                encoding="utf-8")
            recipe.check_preselected_register("language", sel)

    def test_preselected_language_set_without_a_record_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                recipe.check_preselected_register("language", Path(t))
            self.assertIn("cannot be checked", out.getvalue())

    def test_both_selectors_default_the_filter_off(self):
        self.assertIs(select_common_voice.build_parser().parse_args(["--out-dir", "x"]).require_colloquial, False)
        self.assertIs(select_mdc.build_parser().parse_args([]).require_colloquial, False)
        self.assertIs(select_common_voice.build_parser().parse_args(
            ["--out-dir", "x", "--require-colloquial"]).require_colloquial, True)
        self.assertIs(select_mdc.build_parser().parse_args(["--require-colloquial"]).require_colloquial, True)


if __name__ == "__main__":
    unittest.main()
