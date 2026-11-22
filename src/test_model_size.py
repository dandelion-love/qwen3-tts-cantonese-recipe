"""CPU-only tests for [train].model_size: one setting drives every size check in recipe.py, 1b7 stays the default,
and a config with no tts_model_size is refused whatever the setting (Qwen-TTS Studio's converter would read it as 0b6).

    python src/test_model_size.py
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recipe  # noqa: E402


def make_recipe(work: Path, train: dict | None = None, bake: dict | None = None, paths: dict | None = None):
    """A Recipe on the real recipe.toml with [paths].work at `work` and the given tables overridden."""
    r = recipe.Recipe(ROOT / "recipe.toml", dry=False)
    r.paths = {**r.paths, "work": str(work), **(paths or {})}
    r.cfg = json.loads(json.dumps(r.cfg))
    if train is not None:
        r.cfg["train"] = {k: v for k, v in r.cfg["train"].items() if k != "model_size"} | train
    if bake is not None:
        r.cfg["bake"] = bake
    return r


def write_export(work: Path, cfg: dict) -> Path:
    d = work / "train" / "spk" / "final_merged"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


class ModelSizeTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.work = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def convert(self, r):
        with mock.patch.object(recipe.Recipe, "run") as run, mock.patch.object(recipe.Recipe, "record"), \
                contextlib.redirect_stdout(io.StringIO()):
            recipe.stage_convert(r)
        return run

    def test_default_is_1b7(self):
        self.assertEqual(recipe.model_size(make_recipe(self.work, train={})), "1b7")     # key absent
        self.assertEqual(recipe.model_size(recipe.Recipe(ROOT / "recipe.toml", dry=True)), "1b7")  # as shipped

    def test_unknown_size_is_refused(self):
        with self.assertRaises(SystemExit):
            recipe.model_size(make_recipe(self.work, train={"model_size": "0.6B"}))

    def test_convert_default_accepts_1b7_and_refuses_0b6(self):
        write_export(self.work, {"tts_model_size": "1b7", "talker_config": {"spk_id": {"hk_voice_343": 3000,
                                                                                         "hk_voice_777": 3001}}})
        self.assertTrue(self.convert(make_recipe(self.work, train={})).called)
        r = make_recipe(self.work, train={})
        cfg = json.loads((recipe.export_dir(r) / "config.json").read_text(encoding="utf-8"))
        cfg["tts_model_size"] = "0b6"
        (recipe.export_dir(r) / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaises(SystemExit) as e:
            self.convert(r)
        self.assertIn("'1b7'", str(e.exception))

    def test_convert_0b6_setting_accepts_0b6_and_refuses_1b7(self):
        d = write_export(self.work, {"tts_model_size": "0b6", "talker_config": {"spk_id": {"hk_voice_343": 3000}}})
        r = make_recipe(self.work, train={"model_size": "0b6"}, bake={"voices": {"hk_voice_343": 3000}})
        self.assertTrue(self.convert(r).called)
        (d / "config.json").write_text(json.dumps({"tts_model_size": "1b7", "talker_config": {
            "spk_id": {"hk_voice_343": 3000}}}), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.convert(r)

    def test_missing_size_key_is_refused_for_both_settings(self):
        write_export(self.work, {"talker_config": {"spk_id": {"hk_voice_343": 3000}}})
        for size in recipe.MODEL_SIZES:
            r = make_recipe(self.work, train={"model_size": size}, bake={"voices": {"hk_voice_343": 3000}})
            with self.assertRaises(SystemExit) as e:
                self.convert(r)
            self.assertIn("no tts_model_size", str(e.exception))

    def test_train_refuses_a_base_of_the_other_size_before_running(self):
        base = self.work / "base"
        base.mkdir()
        (base / "config.json").write_text(json.dumps({"tts_model_size": "1b7"}), encoding="utf-8")
        data = self.work / "data.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        r = make_recipe(self.work, train={"model_size": "0b6"})
        with mock.patch.object(recipe.Recipe, "run") as run, self.assertRaises(SystemExit) as e:
            recipe.train(r, "lang", base, data, [])
        run.assert_not_called()
        self.assertIn("tts_model_size", str(e.exception))
        self.assertIn("'0b6'", str(e.exception))

    def test_train_refuses_an_init_without_config_json(self):
        data = self.work / "data.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        r = make_recipe(self.work, train={})
        with mock.patch.object(recipe.Recipe, "run") as run, self.assertRaises(SystemExit) as e:
            recipe.train(r, "lang", self.work / "nobase", data, [])
        run.assert_not_called()
        self.assertIn("config.json is missing", str(e.exception))

    def test_train_export_check_reads_the_setting(self):
        data = self.work / "data.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        out = self.work / "train" / "lang" / "final_merged"

        def fake_run(stage, cmd, cwd=None, env=None):
            out.mkdir(parents=True)
            (out / "config.json").write_text(json.dumps({"tts_model_size": "0b6", "talker_config": {}}),
                                             encoding="utf-8")
        base = self.work / "base"
        base.mkdir()
        cm = mock.Mock()
        cm.check_floor.return_value = {"mib": 99999}
        for size, ok in (("0b6", True), ("1b7", False)):
            if out.exists():
                import shutil
                shutil.rmtree(out)
            r = make_recipe(self.work, train={"model_size": size})
            (base / "config.json").write_text(json.dumps({"tts_model_size": size}), encoding="utf-8")
            with mock.patch.object(recipe.Recipe, "run", side_effect=fake_run), \
                    mock.patch.object(recipe, "card_memory", return_value=cm), \
                    mock.patch.object(recipe, "mlflow_run_id", return_value=None):
                if ok:
                    self.assertEqual(recipe.train(r, "lang", base, data, [])["tts_model_size"], "0b6")
                else:
                    with self.assertRaises(SystemExit) as e:
                        recipe.train(r, "lang", base, data, [])
                    self.assertIn("lang export", str(e.exception))

    def test_bake_command_carries_the_size_and_derives_without_a_source(self):
        r = make_recipe(self.work, train={"model_size": "0b6"}, paths={"voice_source": ""},
                        bake={"voices": {"hk_voice_343": 3000},
                              "voice_audio": {"hk_voice_343": ["/clips/a.flac", "/clips/b,c.flac"]}})
        cmd = [str(c) for c in recipe.bake_command(r)]
        self.assertIn("--voice-audio=hk_voice_343=/clips/b,c.flac", cmd)     # one flag per clip, a comma survives
        self.assertEqual(cmd[cmd.index("--model-size") + 1], "0b6")
        self.assertNotIn("--source", cmd)
        self.assertEqual(cmd[cmd.index("--encoder") + 1], str(self.work / "train" / "lang" / "final_merged"))
        self.assertIn("--voice-audio=hk_voice_343=/clips/a.flac", cmd)

    def score_cmd(self, r):
        with mock.patch.object(recipe.Recipe, "run") as run, mock.patch.object(recipe.Recipe, "record"):
            recipe.stage_score(r)
        return [str(c) for c in run.call_args[0][1]]

    def test_score_passes_the_size_and_clips_to_its_bake(self):
        r = make_recipe(self.work, train={"model_size": "0b6"}, paths={"voice_source": ""},
                        bake={"voices": {"hk_voice_343": 3000}, "voice_audio": {"hk_voice_343": ["/clips/a.flac"]}})
        r.dry = True                                   # placeholders for the unset paths; run is mocked anyway
        cmd = self.score_cmd(r)
        self.assertEqual(cmd[cmd.index("--model-size") + 1], "0b6")
        self.assertIn("--voice-audio=hk_voice_343=/clips/a.flac", cmd)
        self.assertNotIn("--voice-source", cmd)

    def test_score_default_is_1b7_from_the_voice_source(self):
        r = make_recipe(self.work, train={}, paths={"voice_source": str(self.work / "release")})
        r.dry = True
        cmd = self.score_cmd(r)
        self.assertEqual(cmd[cmd.index("--model-size") + 1], "1b7")
        self.assertEqual(cmd[cmd.index("--voice-source") + 1], str(self.work / "release"))

    def test_bake_command_default_is_the_1b7_copy(self):
        r = make_recipe(self.work, train={}, paths={"voice_source": str(self.work / "release")})
        cmd = [str(c) for c in recipe.bake_command(r)]
        self.assertEqual(cmd[cmd.index("--model-size") + 1], "1b7")
        self.assertEqual(cmd[cmd.index("--source") + 1], str(self.work / "release"))
        self.assertNotIn("--encoder", cmd)

    def test_bake_command_refuses_a_voice_with_neither_source_nor_clips(self):
        r = make_recipe(self.work, train={"model_size": "0b6"}, paths={"voice_source": ""},
                        bake={"voices": {"hk_voice_343": 3000, "hk_voice_777": 3001},
                              "voice_audio": {"hk_voice_343": ["/clips/a.flac"]}})
        with self.assertRaises(SystemExit):
            recipe.bake_command(r)


if __name__ == "__main__":
    unittest.main()
