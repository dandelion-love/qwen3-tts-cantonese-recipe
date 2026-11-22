#!/usr/bin/env python3
"""CPU tests for src/score_checkpoints.py, with the renderer, the recogniser and MLflow faked: no GPU, no model.

    python src/test_score_checkpoints.py

AC1  every checkpoint is rendered, judged and written to the summary and MLflow (fake), the backend on every row
AC2  with no --scorer nothing extra is imported; with one set, it is imported and its numbers are logged
AC3  one checkpoint scored twice gives the floor, written beside every reading; a no-op (repeat) floor reads zero
AC4  a missing or partial checkpoint is refused by name, before anything renders
and: a step's weights become an export with --init's keys and dtypes (LoRA folded in), a running trainer refuses the
run, and rows from two backends refuse the comparison.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_checkpoints as sc  # noqa: E402

TEXT = "今日天氣好好所以我哋一齊去公園"          # invented for this test
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail if not cond else ''}")
    if not cond:
        FAILS.append(name)


def refuses(fn, *words) -> bool:
    try:
        fn()
    except SystemExit as e:
        return all(w in str(e) for w in words)
    return False


def write_st(p: Path, tensors: dict) -> None:
    from safetensors.torch import save_file
    save_file(tensors, str(p))


def step_dir(root: Path, n: int, ok: bool = True) -> Path:
    import torch
    d = root / f"step-{n}"
    d.mkdir(parents=True)
    write_st(d / "model.safetensors", {"w": torch.ones(2, 2)})
    if ok:
        (d / "resume_state.json").write_text(json.dumps({"optimizer_steps": n}))
    return d


# ---------------------------------------------------------------------------------------------------- fakes
class FakeMlflow:
    def __init__(self):
        self.metrics, self.params = [], {}

    def set_tracking_uri(self, u): self.uri = u
    def set_experiment(self, e): self.exp = e
    def log_params(self, p): self.params.update(p)
    def log_metric(self, k, v, step=0): self.metrics.append((k, v, step))
    def log_dict(self, d, name): self.dict = d

    def start_run(self, run_name=None):
        outer = self

        class Run:
            info = types.SimpleNamespace(run_id="fake-run")
            def __enter__(self): return self
            def __exit__(self, *x): return False
        outer.run_name = run_name
        return Run()


def fake_render(speech_for):
    """A render that writes manifest rows; speech_for(arm, digest) gives the clip's speech seconds."""
    calls = []

    def render(arm, gguf, lines, renders, draws, speaker):
        calls.append(arm)
        d = renders / arm
        (d / "clips").mkdir(parents=True, exist_ok=True)
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in sc.read_jsonl(lines):
                for k in range(draws):
                    cid = f"{r['digest']}-d{k}"
                    (d / "clips" / f"{cid}.flac").write_bytes(b"x")
                    f.write(json.dumps({"clip_id": cid, "digest": r["digest"], "draw": k, "text": r["text"],
                                        "flac": f"clips/{cid}.flac", "speech_s": speech_for(arm, r["digest"]),
                                        "speech_rms": 0.05, "backend": "gguf"}, ensure_ascii=False) + "\n")
    render.calls = calls
    return render


def setup(td: Path, n_lines: int = 6):
    ck = td / "run" / "resume"
    for n in (500, 1000):
        step_dir(ck, n)
    lines = td / "lines.jsonl"
    lines.write_text("".join(json.dumps({"digest": f"L{i}", "text": TEXT}, ensure_ascii=False) + "\n"
                             for i in range(n_lines)), encoding="utf-8")
    init = td / "init"                          # an untrained export: scored as the step-0 base row
    init.mkdir()
    (init / "config.json").write_text("{}")
    import torch
    write_st(init / "model.safetensors", {"w": torch.ones(2, 2)})
    a = sc.parser().parse_args(["--checkpoints", str(ck), "--init", str(init), "--lines", str(lines),
                                "--out", str(td / "out"), "--speaker", "v"])
    a.key_prefix, a.mlflow_uri = "score-t", "sqlite:///unused"
    return a


def run(a, render, proc_root, mlflow=None, judge=lambda dev: (lambda path: TEXT)):
    return sc.score(a, prepare=lambda ck, a: (Path("gguf"), None), render=render, judge_factory=judge,
                    mlflow_mod=mlflow or FakeMlflow(), proc_root=proc_root)


# ---------------------------------------------------------------------------------------------------- tests
def test_refusals(td: Path):
    import torch
    root = td / "refuse"
    good = step_dir(root, 500)
    no_state = step_dir(root, 1000, ok=False)
    tmp = root / "step-1500.tmp"                # whole files inside: only the name says the write is unfinished
    tmp.mkdir()
    write_st(tmp / "model.safetensors", {"w": torch.ones(2, 2)})
    (tmp / "resume_state.json").write_text(json.dumps({"optimizer_steps": 1500}))
    trunc = step_dir(root, 2000)
    b = (trunc / "model.safetensors").read_bytes()
    (trunc / "model.safetensors").write_bytes(b[:-4])
    check("AC4 a complete step-<N> is accepted", sc.checkpoint_problem(good) is None)
    check("AC4 discover lists step-<N>.tmp too, in step order",
          [p.name for p in sc.discover(root)] == ["step-500", "step-1000", "step-1500.tmp", "step-2000"])
    check("AC4 RED: missing is refused by name", refuses(lambda: sc.refuse_bad([root / "step-9"]), "step-9", "missing"))
    check("AC4 RED: an unfinished .tmp is refused by name", refuses(lambda: sc.refuse_bad([tmp]), "step-1500.tmp", "unfinished write"))
    check("AC4 RED: no resume_state.json is refused by name",
          refuses(lambda: sc.refuse_bad([no_state]), "step-1000", "resume_state.json"))
    check("AC4 RED: a truncated safetensors is refused by name",
          refuses(lambda: sc.refuse_bad([trunc]), "step-2000", "truncated"))
    exp = root / "final_merged"
    exp.mkdir()
    (exp / "config.json").write_text("{}")
    check("AC4 RED: an export without weights is refused",
          refuses(lambda: sc.refuse_bad([exp]), "final_merged", "no model"))
    write_st(exp / "model.safetensors", {"w": torch.ones(1)})
    check("AC4 an export with whole weights is accepted", sc.checkpoint_problem(exp) is None)
    # and the run refuses before rendering anything
    a = setup(td / "refuse-run")
    step_dir(Path(a.checkpoints[0]), 1500, ok=False)
    r = fake_render(lambda arm, d: 3.6)
    check("AC4 the run refuses a partial checkpoint by name before any render",
          refuses(lambda: run(a, r, td / "noproc"), "step-1500") and r.calls == [])


def test_scoring_floor_and_mlflow(td: Path):
    a = setup(td / "ac1")
    r = fake_render(lambda arm, d: 3.6)
    ml = FakeMlflow()
    s = run(a, r, td / "noproc", ml)
    labels = [c["label"] for c in s["checkpoints"]]
    check("AC1 every saved checkpoint is scored, in step order", labels == ["base", "step-500", "step-1000"], str(labels))
    check("AC1 the floor checkpoint (the last) is rendered a second time", r.calls == ["base", "step-500", "step-1000", "step-1000~floor"],
          str(r.calls))
    m = s["checkpoints"][1]["measures"]
    check("AC1 CER, runaway, silence and skipped-text rates per checkpoint",
          set(m) == set(sc.MEASURES) and m["cer_mean"]["value"] == 0.0 and m["runaway_rate"]["value"] == 0.0)
    check("AC1 the backend is recorded on every reading", all(c["backend"] == "gguf" for c in s["checkpoints"]))
    written = json.loads((a.out / "summary.json").read_text(encoding="utf-8"))
    check("AC1 the JSON summary is written", written["checkpoints"][2]["label"] == "step-1000")
    rows = sc.read_jsonl(a.out / "clips.jsonl")
    check("AC1 every clip row carries its backend", rows and all(x["backend"] == "gguf" for x in rows))
    keys = {(k, step) for k, v, step in ml.metrics}
    check("AC1 MLflow gets each measure at step N", ("runaway_rate", 500) in keys and ("cer_mean", 1000) in keys)
    check("AC3 MLflow gets the floor beside every measure at every step",
          all((f"{m}_floor", n) in keys for m in sc.MEASURES for n in (500, 1000)))
    check("AC3 a second scoring of identical audio reads a zero floor (a no-op reads as no change)",
          all(v == 0.0 for v in s["floor"]["difference"].values()), str(s["floor"]))
    # a reseeded second scoring where the draw differs: the floor is the difference
    b = setup(td / "ac3")
    r2 = fake_render(lambda arm, d: 12.0 if arm.endswith("~floor") and d == "L0~floor" else 3.6)
    s2 = run(b, r2, td / "noproc")
    f = s2["floor"]["difference"]["runaway_rate"]
    check("AC3 RED: a second scoring with one runaway in six reads a floor of 1/6", abs(f - 1 / 6) < 1e-3, str(f))
    check("AC3 the floor is written beside every reading",
          all(c["measures"]["runaway_rate"]["floor"] == f for c in s2["checkpoints"]))
    # a re-run renders and judges nothing again
    n = []
    run(b, r2, td / "noproc", judge=lambda dev: (n.append(1), lambda p: TEXT)[1])
    check("a re-run renders nothing and judges nothing", r2.calls == ["base", "step-500", "step-1000", "step-1000~floor"] and not n)


def test_hook(td: Path):
    mod = "fake_private_scorer_mod"
    hdir = td / "hookmod"
    hdir.mkdir()
    (hdir / f"{mod}.py").write_text("CALLS = []\ndef score(label, step, clips):\n    CALLS.append(label)\n"
                                    "    return {'extra': float(len(clips))}\n", encoding="utf-8")
    sys.path.insert(0, str(hdir))
    a = setup(td / "nohook")
    run(a, fake_render(lambda arm, d: 3.6), td / "noproc")
    check("AC2 with no scorer set, nothing extra is imported", mod not in sys.modules)
    b = setup(td / "hook")
    b.scorer = f"{mod}:score"
    ml = FakeMlflow()
    s = run(b, fake_render(lambda arm, d: 3.6), td / "noproc", ml)
    check("AC2 with a scorer set, it is imported and called per scoring",
          mod in sys.modules and sys.modules[mod].CALLS == ["base", "step-500", "step-1000", "step-1000~floor"])
    check("AC2 its numbers are logged with a floor beside them",
          s["checkpoints"][1]["hook"]["extra"] == {"value": 6.0, "floor": 0.0, "vs_base": 0.0}
          and ("hook/extra", 500, ) in {(k, st) for k, v, st in ml.metrics})
    c = setup(td / "badhook")
    c.scorer = "no_colon"
    check("AC2 RED: a malformed scorer refuses before anything runs",
          refuses(lambda: run(c, fake_render(lambda arm, d: 3.6), td / "noproc"), "module:function"))


def test_guard(td: Path):
    proc = td / "proc"
    (proc / "4242").mkdir(parents=True)
    (proc / "4242" / "cmdline").write_bytes(b"python\0trainer/sft_12hz_lora_mlflow.py\0--train_jsonl\0x")
    a = setup(td / "guard")
    r = fake_render(lambda arm, d: 3.6)
    check("RED: a running trainer refuses the run, naming the process", refuses(lambda: run(a, r, proc), "pid 4242")
          and r.calls == [])
    a.beside_training = True
    s = run(a, r, proc)
    check("--beside-training proceeds and records it", s["guard"]["beside_training"] and s["guard"]["training_processes"])
    check("no trainer: the guard passes", sc.guard(False, td / "noproc")["training_processes"] == [])


def test_backend_mix():
    check("RED: rows from two backends refuse the comparison",
          refuses(lambda: sc.one_backend([{"backend": "gguf"}, {"backend": "torch"}]), "more than one backend"))
    check("one backend passes", sc.one_backend([{"backend": "gguf"}] * 3) == "gguf")


def test_materialize(td: Path):
    import torch
    init = td / "mat-init"
    init.mkdir()
    (init / "config.json").write_text('{"tts_model_size": "x"}')
    base = {"talker.a.weight": torch.zeros(2, 3, dtype=torch.bfloat16), "talker.b": torch.ones(2, dtype=torch.bfloat16)}
    write_st(init / "model.safetensors", base)
    # a full fine-tune step: natural keys, float32
    full = td / "mat" / "step-500"
    full.mkdir(parents=True)
    write_st(full / "model.safetensors", {"talker.a.weight": torch.full((2, 3), 0.5), "talker.b": torch.full((2,), 2.0),
                                          "speaker_encoder.x": torch.ones(1)})
    (full / "resume_state.json").write_text('{"optimizer_steps": 500}')
    out = sc.materialize(full, init, td / "mat-out" / "step-500", None)
    from safetensors.torch import load_file
    got = load_file(str(out / "model.safetensors"))
    check("a full fine-tune step becomes an export with --init's keys and dtypes",
          set(got) == set(base) and got["talker.a.weight"].dtype == torch.bfloat16
          and float(got["talker.a.weight"][0, 0]) == 0.5 and (out / "config.json").is_file())
    # a LoRA step: PEFT keys, folded at alpha / rank
    lora = td / "mat" / "step-1000"
    lora.mkdir(parents=True)
    A, B = torch.ones(1, 3), torch.full((2, 1), 2.0)
    write_st(lora / "model.safetensors", {"base_model.model.talker.a.base_layer.weight": torch.zeros(2, 3),
                                          "base_model.model.talker.a.lora_A.default.weight": A,
                                          "base_model.model.talker.a.lora_B.default.weight": B,
                                          "base_model.model.talker.b": torch.ones(2)})
    (lora / "resume_state.json").write_text('{"optimizer_steps": 1000}')
    got = load_file(str(sc.materialize(lora, init, td / "mat-out" / "step-1000", 2.0) / "model.safetensors"))
    check("a LoRA step is folded in at scale alpha/rank (0 + 2 * B @ A = 4)",
          set(got) == set(base) and float(got["talker.a.weight"][1, 2]) == 4.0, str(got.get("talker.a.weight")))
    check("RED: a LoRA step without a scale refuses",
          refuses(lambda: sc.materialize(lora, init, td / "mat-out" / "x", None), "--lora-rank"))
    bad = td / "mat" / "step-1500"
    bad.mkdir(parents=True)
    write_st(bad / "model.safetensors", {"talker.a.weight": torch.zeros(2, 3)})
    (bad / "resume_state.json").write_text('{"optimizer_steps": 1500}')
    check("RED: a step missing a key of --init refuses, naming it",
          refuses(lambda: sc.materialize(bad, init, td / "mat-out" / "y", None), "1 missing", "talker.b"))


def test_fold(td: Path):
    t2s = sc.load_t2s()
    trad = "我哋嘅關係好好所以一齊體驗下"               # invented; 關係 and 體驗 are written differently in simplified
    simp = t2s.convert(trad)
    row = {"clip_id": "a", "text": trad, "speech_s": 3.0, "speech_rms": 0.05, "backend": "gguf"}
    m = sc.clip_measures(row, simp, sc.RULES, t2s)
    check("fold: the plant is real (the same words in simplified score as error unfolded)", m["cer_unfolded"] > 0, str(m))
    check("fold RED: a simplified transcript of a traditional line is CER 0 and no skip",
          m["cer"] == 0.0 and not m["skipped"], str(m))
    lost = sc.clip_measures(row, simp[:3] + simp[9:], sc.RULES, t2s)
    check("fold: a real skip in a simplified transcript is still found", lost["skipped"] and lost["cer"] > 0, str(lost))
    # Four script-differing characters in a row: unfolded, the aligner pairs the transcript's simplified run with the
    # reference's later simplified run and reads the traditional one as five characters lost. Folded, the only loss is
    # the one character the recogniser really dropped, under the run floor.
    mixed = "我哋嘅關係體驗好关系体验"                       # invented; the reference itself mixes the two scripts
    heard = t2s.convert(mixed).replace("好", "")
    import skipped_text as sk
    check("fold: the plant is real (unfolded, a run of 4+ script-differing characters reads as skipped text)",
          bool(sk.lost_runs(mixed, heard, sc.RULES["skip_min_run"])))
    check("fold RED: folded, the same transcript is no skipped text",
          not sc.clip_measures({**row, "text": mixed}, heard, sc.RULES, t2s)["skipped"])
    saved = sys.modules.get("opencc")
    sys.modules["opencc"] = None                    # makes `import opencc` raise ImportError
    try:
        check("fold RED: without OpenCC the scorer refuses rather than score unfolded",
              refuses(sc.load_t2s, "OpenCC"))
        a = setup(td / "no-opencc")
        r = fake_render(lambda arm, d: 3.6)
        check("fold: the run refuses without OpenCC before any render", refuses(lambda: run(a, r, td / "noproc"),
                                                                                "OpenCC") and r.calls == [])
    finally:
        if saved is None:
            sys.modules.pop("opencc", None)
        else:
            sys.modules["opencc"] = saved


def test_base_rows_and_nulls(td: Path):
    a = setup(td / "base")
    ml = FakeMlflow()
    r = fake_render(lambda arm, d: 12.0 if arm == "base" and d == "L0" else 3.6)
    s = run(a, r, td / "noproc", ml)
    st = s["checkpoints"][1]["measures"]["runaway_rate"]
    check("base: the untrained --init is scored as the step-0 row", s["checkpoints"][0]["label"] == "base"
          and s["checkpoints"][0]["step"] == 0 and s["base"]["label"] == "base")
    check("base: every reading carries its difference from the base row", abs(st["vs_base"] - (-1 / 6)) < 1e-3
          and all(m["vs_base"] is not None for c in s["checkpoints"] for m in c["measures"].values()), str(st))
    check("base: MLflow gets <measure>_vs_base", ("runaway_rate_vs_base", 500) in {(k, x) for k, v, x in ml.metrics})
    w = json.loads((a.out / "summary.json").read_text(encoding="utf-8"))
    check("summary.json marks the rules uncalibrated", w["rules_status"].startswith("uncalibrated"))
    check("summary.json records the disk each checkpoint holds", w["disk"]["total_bytes"] > 0
          and set(w["disk"]["bytes"]) == {"base", "step-500", "step-1000"})
    b = setup(td / "nobase")
    b.no_base = True
    s = run(b, fake_render(lambda arm, d: 3.6), td / "noproc")
    check("--no-base: no base row, the reason recorded, vs_base null",
          [c["label"] for c in s["checkpoints"]] == ["step-500", "step-1000"] and s["base"]["reason"] == "--no-base"
          and s["checkpoints"][0]["measures"]["cer_mean"]["vs_base"] is None)
    mod = "fake_nan_scorer_mod"
    hdir = td / "nanmod"
    hdir.mkdir()
    (hdir / f"{mod}.py").write_text("def score(label, step, clips):\n"
                                    "    return {'odd': float('nan')} if label.endswith('~floor') else {'odd': 1.0}\n",
                                    encoding="utf-8")
    sys.path.insert(0, str(hdir))
    c = setup(td / "nan")
    c.scorer = f"{mod}:score"
    s = run(c, fake_render(lambda arm, d: 3.6), td / "noproc")

    def no_nan(x):
        raise ValueError(f"{x} in summary.json")
    try:
        w = json.loads((c.out / "summary.json").read_text(encoding="utf-8"), parse_constant=no_nan)
        clean = True
    except ValueError:
        clean = False
    check("RED: a hook NaN never reaches summary.json: null, with the reason recorded",
          clean and w["floor"]["hook"]["odd"] is None and w["nulls"], str(w.get("nulls") if clean else "NaN written"))


def test_speaker_phase(td: Path):
    import torch
    a = setup(td / "spk-fallback")
    final = td / "spk-fallback" / "run" / "final_merged"
    final.mkdir(parents=True)
    (final / "config.json").write_text('{"tts_model_type": "custom_voice", "talker_config": {"spk_id": {"v": 3000}}}')
    write_st(final / "model.safetensors", {"w": torch.ones(2, 2)})
    a.checkpoints = [td / "spk-fallback" / "run" / "no-resume"]
    a.checkpoint = [str(final)]
    s = run(a, fake_render(lambda arm, d: 3.6), td / "noproc")
    check("spk: no step-<N> at all falls back to the named final_merged",
          [c["label"] for c in s["checkpoints"]] == ["base", "final"])
    a.checkpoint = []
    check("spk: without --own-voice the base row carries no voice caveat",
          "voice_caveat" not in s["base"])
    a.own_voice = ["v=3000"]
    a.checkpoint = [str(final)]
    s = run(a, fake_render(lambda arm, d: 3.6), td / "noproc")
    check("spk: with --own-voice the base row carries the voice caveat", "borrowed" in s["base"].get("voice_caveat", ""))
    a.own_voice = []
    a.checkpoint = []
    check("spk RED: with nothing else named, a missing resume/ still refuses",
          refuses(lambda: run(a, fake_render(lambda arm, d: 3.6), td / "noproc"), "missing"))
    # a speaker-phase step is baked with its OWN running mean
    init = td / "own" / "init"
    init.mkdir(parents=True)
    (init / "config.json").write_text('{"tts_model_type": "base", "talker_config": {}}')
    key = "talker.model.codec_embedding.weight"
    write_st(init / "model.safetensors", {key: torch.zeros(4, 3, dtype=torch.bfloat16)})
    step = td / "own" / "step-500"
    step.mkdir()
    write_st(step / "model.safetensors", {key: torch.zeros(4, 3)})
    (step / "resume_state.json").write_text('{"optimizer_steps": 500}')
    write_st(step / "voice_means.safetensors", {"mean.v": torch.tensor([[1.0, 2.0, 3.0]])})
    out = sc.materialize(step, init, td / "own" / "out" / "step-500", None, {"v": 2})
    from safetensors.torch import load_file
    w = load_file(str(out / "model.safetensors"))[key]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    check("spk: a step is scored in the voice it trained (its running mean at the named row)",
          w[2].float().tolist() == [1.0, 2.0, 3.0] and w[1].float().tolist() == [0.0, 0.0, 0.0], str(w))
    check("spk: the scoring copy is custom_voice naming that voice",
          cfg["tts_model_type"] == "custom_voice" and cfg["talker_config"]["spk_id"] == {"v": 2})
    check("spk: --init is not changed", json.loads((init / "config.json").read_text())["tts_model_type"] == "base")
    (step / "voice_means.safetensors").unlink()
    check("spk RED: a step without voice_means is refused by name",
          refuses(lambda: sc.materialize(step, init, td / "own" / "out" / "x", None, {"v": 2}), "step-500",
                  "voice_means"))


def test_scoring_copy_only(td: Path):
    import torch
    lang = td / "copy" / "lang" / "final_merged"       # a language phase's export: base type, no voice
    lang.mkdir(parents=True)
    original = '{"tts_model_type": "base", "talker_config": {}}'
    (lang / "config.json").write_text(original)
    write_st(lang / "model.safetensors", {"w": torch.ones(1)})
    converted = []

    bakes = []

    def fake_bake(target, source, voices, **kw):
        bakes.append(kw)
        cfg = json.loads((target / "config.json").read_text())
        cfg["talker_config"]["spk_id"] = voices
        (target / "config.json").write_text(json.dumps(cfg))
        return {"baked": voices}
    fakes = {"gguf_backend": types.SimpleNamespace(ensure_converted=lambda src, key, quant: converted.append(src) or Path("g")),
             "bake_voices": types.SimpleNamespace(bake=fake_bake)}
    saved = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        a = sc.parser().parse_args(["--lines", "x", "--out", str(td / "copy" / "out"), "--speaker", "v",
                                    "--bake", "v=3000", "--voice-source", str(td)])
        a.key_prefix = "k"
        g, copy = sc.prepare_gguf({"label": "final", "path": lang, "step": None, "kind": "export"}, a)
        cfg = json.loads((copy / "config.json").read_text())
        check("base type: the scoring copy is set custom_voice with the baked voice",
              cfg["tts_model_type"] == "custom_voice" and cfg["talker_config"]["spk_id"] == {"v": 3000}
              and json.loads((copy / "scoring.json").read_text())["tts_model_type_was"] == "base")
        check("base type: the trained export is never changed", (lang / "config.json").read_text() == original
              and converted == [copy])
        spk = td / "copy" / "spk" / "final_merged"           # already carries the voice: rendered as it is
        spk.mkdir(parents=True)
        (spk / "config.json").write_text('{"tts_model_type": "custom_voice", "talker_config": {"spk_id": {"v": 3000}}}')
        g, copy = sc.prepare_gguf({"label": "spkfinal", "path": spk, "step": None, "kind": "export"}, a)
        check("an export carrying the voice is converted as it is, with no copy", copy is None and converted[-1] == spk)
        check("the bake gets the 1b7 default size and no clips", bakes[0].get("model_size") == "1b7"
              and not bakes[0].get("audio"))
        a = sc.parser().parse_args(["--lines", "x", "--out", str(td / "copy" / "out"), "--speaker", "v",
                                    "--bake", "v=3000", "--model-size", "0b6", "--init", str(td / "init"),
                                    "--voice-audio", "v=/clips/a,b.flac", "--voice-audio", "v=/clips/c.flac"])
        a.key_prefix = "k"
        sc.prepare_gguf({"label": "final6", "path": lang, "step": None, "kind": "export"}, a)
        check("a 0b6 setting reaches the bake, with each clip and --init as the encoder",
              bakes[-1] == {"model_size": "0b6", "encoder": td / "init",
                            "audio": {"v": ["/clips/a,b.flac", "/clips/c.flac"]}})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        (td / "noproc").mkdir()
        test_refusals(td)
        test_scoring_floor_and_mlflow(td)
        test_hook(td)
        test_guard(td)
        test_backend_mix()
        test_materialize(td)
        test_fold(td)
        test_base_rows_and_nulls(td)
        test_speaker_phase(td)
        test_scoring_copy_only(td)
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)}: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
