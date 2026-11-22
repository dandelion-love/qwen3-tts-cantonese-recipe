#!/usr/bin/env python3
"""Score the checkpoints a training phase saved, after it, from disk: training is never paused for a measurement.

WHY AFTERWARDS. A full fine-tune that is paused and resumed is not the same run (the trainer's resume notes: AdamW
moments and the fp32 masters are not restored), and a render beside a training phase on a 12 GB card roughly triples
its step time. So the trainer writes step-<N> directories (--checkpoint_every_steps; pass --keep_step_checkpoints or
only the newest is kept) and this scores them once it is done. By default it REFUSES while a trainer process is
running (--beside-training overrides, and the summary records that it did).

WHAT IT DOES, per checkpoint, every one through the same path:

    1. refuse, by name, any checkpoint that is missing or partial (a step-<N>.tmp left by an unfinished write, a
       step-<N> without model.safetensors or resume_state.json, a truncated safetensors file), before any work
    2. a step-<N> becomes a loadable export: the --init model's files with the step's weights (LoRA folded in at
       alpha / rank), every key and dtype checked against --init; an export (config.json) is used as it is
    3. give it a voice, in a SCORING COPY only (the training output is never changed): a speaker phase's step gets
       its OWN running mean (--own-voice, from the voice_means the trainer saved beside it), so it is scored in the
       voice it trained; a language-phase step or an export without the voice gets the fixed --bake row from
       --voice-source (src/bake_voices.py). A copy with a voice is marked tts_model_type custom_voice (recorded in
       its scoring.json) so the renderer takes --speaker on it; an export that already carries the voice is used as
       it is
    4. convert to GGUF (gguf_backend.ensure_converted, cached and fingerprinted) and render the fixed lines through
       src/render_short.py: the same lines, the same per-clip seeds (sha256 of digest and draw), no edge trimming
    5. after every render, judge each clip with Qwen3-ASR (the release gate's recogniser, src/steps.py)

and writes, per checkpoint: mean and median CER and the RUNAWAY, SILENCE and SKIPPED-TEXT rates, to
<out>/summary.json and to MLflow (step = N). CER and skipped text are on the release gate's script fold
(release_gate.folded_scores: both sides to simplified, CJK only), so a recogniser writing the same words in the other
script is not an error; it refuses without OpenCC rather than fall back, and the unfolded CER is kept beside it,
reported only.

THE BASE ROW. The untrained --init (the model the phase started from) is scored first as step 0, "base", on the same
lines, seeds, voice and judge (--no-base skips it), and every reading carries vs_base, its value minus the base's: the
untrained model has runaways of its own, so a rate means something only against it. The step sizes on disk and their
total are printed before any work. The backend is recorded on every clip row, and a run whose rows came
from more than one backend is refused: a comparison across backends measures the backend.

THE NO-OP FLOOR. One checkpoint (--floor-checkpoint, default the last) is scored a second time and the absolute
difference of every measure between the two scorings is the floor, written beside every reading (summary: each
measure as {"value", "floor"}; MLflow: <measure> and <measure>_floor at every step). --floor-mode reseed (default)
renders the second scoring at fresh seeds, so the floor is draw-to-draw noise: the size a difference between two
checkpoints must exceed before it means anything. --floor-mode repeat uses the same seeds; on GGUF that audio is
byte-identical, so it reads zero unless the judge itself moves (a determinism check, not a noise floor).

THE MEASURES REPORT; THEY NEVER CHOOSE. A rate may veto a checkpoint on a functional failure (non-termination,
silence, lost text); nothing here picks one. The clip rules (RULES, overridable with --rules) are:

    runaway   speech span (seconds from first to last sound) per CJK character of the line above runaway_s_per_char
    silence   speech span under silence_speech_s, or the speech span's RMS under silence_rms
    skipped   a run of skip_min_run or more of the line's CJK characters the recogniser did not hear
              (src/skipped_text.py)

`--self-test` shows each rule firing on a planted defect and reading nothing on a faithful clip; the thresholds are
UNCALIBRATED against listening (summary.json says so in rules_status), so read the rates against the base row and the
floor, as functional checks, not quality. summary.json never holds NaN: a value that is not a finite number is null,
and the reason is listed under "nulls".

THE SCORER HOOK. --scorer module:function adds measures of your own: the module is imported only when the flag is
set (with none set, nothing extra is imported, tested), and the function is called once per scored rendering as
fn(label=..., step=..., clips=[{clip_id, text, flac, hyp, ...}]) -> {name: number}. Its numbers are logged as
hook/<name>, beside the floor of the floor checkpoint's two scorings; they are reported, never used to choose.

THE LINES ARE YOURS. This repository ships no text: --lines is a JSONL of {"digest", "text"} (the release gate's
held-out lines are the usual choice; recipe.toml [score].lines names the [paths] entry).

    python src/score_checkpoints.py --checkpoints RUN/lang/resume --init BASE --lines LINES.jsonl --out OUT \\
        --speaker hk_voice_343 --bake hk_voice_343=3000 --voice-source RELEASE_MERGED
    python src/score_checkpoints.py --self-test          (CPU, seconds; the full test is src/test_score_checkpoints.py)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STEP_RE = re.compile(r"^step-(\d+)(\.tmp)?$")
TRAINER_PATTERNS = ("sft_12hz_lora_mlflow.py",)
RULES = {"runaway_s_per_char": 0.6, "silence_speech_s": 0.3, "silence_rms": 0.003, "skip_min_run": 4}
MEASURES = ("cer_mean", "cer_median", "runaway_rate", "silence_rate", "skipped_rate", "cer_unfolded_mean")
RULES_STATUS = ("uncalibrated: each rule is shown to fire on a synthetic plant (--self-test) and not calibrated "
                "against listening; read the rates against the base row and the floor, as functional checks")
BACKEND = "gguf"
JUDGE = "Qwen/Qwen3-ASR-1.7B, language Cantonese"


# ---------------------------------------------------------------------------------------------------- refusals
def safetensors_problem(p: Path) -> str | None:
    """None if `p` is a whole safetensors file: its header parses and every tensor's bytes are inside the file."""
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return f"{p.name} is {size} bytes, shorter than a safetensors header"
            n = struct.unpack("<Q", raw)[0]
            if n > size - 8:
                return f"{p.name}: header of {n} bytes runs past the end of the file ({size} bytes)"
            head = json.loads(f.read(n))
    except (OSError, ValueError) as e:
        return f"{p.name} is unreadable ({type(e).__name__}: {e})"
    end = max((v["data_offsets"][1] for k, v in head.items() if k != "__metadata__"), default=0)
    if 8 + n + end > size:
        return f"{p.name} is truncated: its tensors need {8 + n + end} bytes and the file has {size}"
    return None


def checkpoint_problem(p: Path) -> str | None:
    """Why `p` cannot be scored, or None. A step-<N> needs model.safetensors and resume_state.json (the trainer writes
    both into step-<N>.tmp and renames it whole); an export needs config.json and its model*.safetensors."""
    if not p.exists():
        return "missing"
    if not p.is_dir():
        return "not a directory"
    if p.name.endswith(".tmp"):
        return "partial: an unfinished write (.tmp); the trainer renames it only when it is complete"
    if (p / "config.json").is_file():
        shards = sorted(p.glob("model*.safetensors"))
        if not shards:
            return "partial: config.json but no model*.safetensors"
        for s in shards:
            why = safetensors_problem(s)
            if why:
                return f"partial: {why}"
        return None
    if not (p / "model.safetensors").is_file():
        return "partial: no model.safetensors (and no config.json, so it is not an export either)"
    if not (p / "resume_state.json").is_file():
        return "partial: no resume_state.json"
    try:
        int(json.loads((p / "resume_state.json").read_text(encoding="utf-8"))["optimizer_steps"])
    except (ValueError, KeyError, TypeError) as e:
        return f"partial: resume_state.json is unreadable ({type(e).__name__})"
    why = safetensors_problem(p / "model.safetensors")
    return f"partial: {why}" if why else None


def discover(root: Path, required: bool = True) -> list[Path]:
    """Every step-<N> (and step-<N>.tmp, so it is refused by name rather than skipped) under `root`, by N. With
    `required` False (another checkpoint is named, e.g. the phase's final_merged), an absent or empty `root` is no
    checkpoint rather than a refusal: a short phase may end before its first --checkpoint_every_steps."""
    if not root.is_dir():
        if not required:
            print(f"NOTE {root}: missing; scoring the named checkpoints only", flush=True)
            return []
        raise SystemExit(f"refusing: {root}: missing (no checkpoint directory)")
    found = [d for d in root.iterdir() if STEP_RE.match(d.name)]
    if not found and required:
        raise SystemExit(f"refusing: {root} holds no step-<N> directory (did the phase run with "
                         f"--checkpoint_every_steps?)")
    return sorted(found, key=lambda d: (int(STEP_RE.match(d.name).group(1)), d.name))


def refuse_bad(paths: list[Path]) -> None:
    bad = [(p, why) for p in paths for why in [checkpoint_problem(p)] if why]
    if bad:
        raise SystemExit("refusing to score: " + "; ".join(f"{p.name}: {why} ({p})" for p, why in bad)
                         + ". Name the complete checkpoints with --checkpoint to score only those.")


def checkpoint_entry(spec: str) -> dict:
    """PATH or PATH@STEP. A step-<N> carries its own step; an export takes @STEP or is logged unnumbered."""
    path, _, step = spec.rpartition("@") if "@" in Path(spec).name else (spec, "", "")
    p = Path(path).expanduser()
    m = STEP_RE.match(p.name)
    n = int(step) if step else (int(m.group(1)) if m and not m.group(2) else None)
    label = p.name if p.name != "final_merged" else "final"
    return {"label": label, "path": p, "step": n, "kind": "export" if (p / "config.json").is_file() else "step"}


# ---------------------------------------------------------------------------------------------------- the trainer guard
def training_processes(patterns=TRAINER_PATTERNS, proc_root: Path = Path("/proc")) -> tuple[list[str], str]:
    """Command lines of running trainer processes, and how they were found ("unchecked" if they could not be)."""
    hits = []
    if proc_root.is_dir():
        for d in proc_root.iterdir():
            if not d.name.isdigit() or int(d.name) == os.getpid():
                continue
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            except OSError:
                continue
            if any(p in cmd for p in patterns):
                hits.append(f"pid {d.name}: {cmd[:200]}")
        return hits, "proc"
    try:
        import psutil
    except ImportError:
        return [], "unchecked"
    for pr in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(pr.info.get("cmdline") or [])
        if pr.info["pid"] != os.getpid() and any(p in cmd for p in patterns):
            hits.append(f"pid {pr.info['pid']}: {cmd[:200]}")
    return hits, "psutil"


def guard(beside_training: bool, proc_root: Path = Path("/proc")) -> dict:
    hits, method = training_processes(proc_root=proc_root)
    if hits and not beside_training:
        raise SystemExit("refusing: a trainer is running, and a render beside training slows it about threefold "
                         "on a 12 GB card. Score after it ends, or pass --beside-training.\n  " + "\n  ".join(hits))
    if method == "unchecked":
        print("WARNING: could not list processes (no /proc, no psutil); a running trainer would not be seen",
              flush=True)
    return {"method": method, "training_processes": hits, "beside_training": bool(hits) and beside_training}


# ---------------------------------------------------------------------------------------------------- step -> export
_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
           "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def init_dtypes(init: Path) -> dict:
    """{key: torch dtype name} of the --init model, read from its headers without loading a tensor."""
    out = {}
    for f in sorted(init.glob("model*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        out.update({k: _DTYPES.get(v["dtype"], v["dtype"]) for k, v in head.items() if k != "__metadata__"})
    if not out:
        raise SystemExit(f"refusing: {init} holds no model*.safetensors")
    return out


def natural_key(k: str, known) -> str:
    """The export's name for a training state key: the PEFT wrapper prefix and base_layer removed, as the trainer's
    own export (clean_merged_state_dict) does, choosing the form --init actually has."""
    for c in (k, k.removeprefix("base_model.model."), k.removeprefix("base_model.model.").removeprefix("model.")):
        c = c.replace(".base_layer.", ".")
        if c in known:
            return c
    return k.removeprefix("base_model.model.").replace(".base_layer.", ".")


def merged_state(sd: dict, dtypes: dict, scale: float | None) -> dict:
    """The step's weights under --init's keys and dtypes, any LoRA folded in (W + scale * B @ A). Refuses a missing
    or unexpected key by count and example: a scoring of a model that loaded 90% of its weights is no scoring."""
    import torch
    has_lora = any(".lora_A." in k for k in sd)
    if has_lora and scale is None:
        raise SystemExit("refusing: a LoRA step checkpoint needs --lora-rank and --lora-alpha to fold it in")
    out = {}
    for k, v in sd.items():
        if ".lora_" in k:
            continue
        if has_lora and ".base_layer." in k:
            pre = k.split(".base_layer.")[0]
            a, b = sd.get(f"{pre}.lora_A.default.weight"), sd.get(f"{pre}.lora_B.default.weight")
            if a is not None and b is not None and k.endswith(".base_layer.weight"):
                v = (v.float() + scale * (b.float() @ a.float()))
        nk = natural_key(k, dtypes)
        if nk not in dtypes and nk.startswith("speaker_encoder."):
            continue                       # an export without the encoder: the trainer drops it the same way
        out[nk] = v.detach().to("cpu").to(getattr(torch, dtypes[nk])) if nk in dtypes else v
    missing, extra = sorted(set(dtypes) - set(out)), sorted(set(out) - set(dtypes))
    if missing or extra:
        raise SystemExit(f"refusing: the step's weights do not match --init: {len(missing)} missing "
                         f"(e.g. {missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")
    return out


def source_stamp(step_dir: Path) -> dict:
    st = (step_dir / "model.safetensors").stat()
    return {"source": str(step_dir.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "resume_state_sha256": hashlib.sha256((step_dir / "resume_state.json").read_bytes()).hexdigest()}


def codec_key(sd: dict) -> str:
    hits = [k for k in sd if k.endswith("codec_embedding.weight")]
    if len(hits) != 1:
        raise SystemExit(f"refusing: {len(hits)} codec_embedding tables in the weights, exactly one expected")
    return hits[0]


def bake_own_voice(sd: dict, step_dir: Path, voices: dict) -> dict:
    """Each voice's row set to the running mean the phase had reached at this step (voice_means.safetensors, written
    by the trainer beside the weights), as the trainer's own export bakes it: a speaker phase is scored in the voice
    it trained, never a borrowed one. Refuses by name when the step carries no mean for a voice."""
    from safetensors.torch import load_file
    f = step_dir / "voice_means.safetensors"
    if not f.is_file():
        raise SystemExit(f"refusing: {step_dir.name}: no voice_means.safetensors, so its own voice cannot be baked "
                         f"({step_dir})")
    means = load_file(str(f))
    key = codec_key(sd)
    for name, row in voices.items():
        t = means.get(f"mean.{name}")
        if t is None:
            raise SystemExit(f"refusing: {step_dir.name}: no running mean for voice {name!r} ({sorted(means)})")
        if row >= sd[key].shape[0]:
            raise SystemExit(f"refusing: row {row} for {name!r} is past the codec table ({sd[key].shape[0]} rows)")
        sd[key][row] = t.reshape(-1).to(sd[key].dtype)
    return {"own_voice": voices, "from": f.name}


def set_custom_voice(d: Path, voices: dict | None = None) -> dict:
    """Mark a SCORING COPY custom_voice, so the renderer takes --speaker on it, with spk_id / spk_is_dialect for
    `voices` if given (else the spk_id already there). Never called on a training output: only on copies under
    <out>/exports. Returns what changed, which also goes into the copy's scoring.json."""
    p = d / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    was = cfg.get("tts_model_type")
    tc = cfg.setdefault("talker_config", {})
    if voices:
        tc["spk_id"] = dict(voices)
    if not tc.get("spk_id"):
        raise SystemExit(f"refusing: {d} names no speaker (talker_config.spk_id); nothing to render with --speaker")
    tc["spk_is_dialect"] = {n: bool((tc.get("spk_is_dialect") or {}).get(n, False)) for n in tc["spk_id"]}
    cfg["tts_model_type"] = "custom_voice"
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rec = {"tts_model_type_was": was, "tts_model_type": "custom_voice", "spk_id": tc["spk_id"],
           "note": "set in this scoring copy only; the training output is unchanged"}
    (d / "scoring.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rec


def materialize(step_dir: Path, init: Path, dest: Path, scale: float | None, own_voice: dict | None = None) -> Path:
    """An export directory at `dest`: --init's files, the step's weights (and, with `own_voice`, the step's own voice
    rows and a custom_voice config). Written to dest.tmp and renamed whole."""
    from safetensors.torch import load_file, save_file
    stamp = {**source_stamp(step_dir), "init": str(init.resolve()), "scale": scale, "own_voice": own_voice}
    if (dest / "materialized.json").is_file() and json.loads((dest / "materialized.json").read_text()) == stamp:
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(init, tmp, ignore=shutil.ignore_patterns("model*.safetensors", "model.safetensors.index.json",
                                                             "materialized.json", "scoring.json"))
    sd = merged_state(load_file(str(step_dir / "model.safetensors")), init_dtypes(init), scale)
    baked = bake_own_voice(sd, step_dir, own_voice) if own_voice else None
    save_file(sd, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    if own_voice:
        set_custom_voice(tmp, own_voice)
    (tmp / "materialized.json").write_text(json.dumps({**stamp, "baked": baked} if baked else stamp, indent=1) + "\n",
                                           encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


def parse_voices(specs: list[str]) -> dict:
    out = {}
    for x in specs or []:
        name, _, row = x.partition("=")
        if not (name and row.isdigit()):
            raise SystemExit(f"refusing: {x!r} is not NAME=ROW")
        out[name] = int(row)
    return out


def speakers_of(d: Path) -> dict:
    return json.loads((d / "config.json").read_text(encoding="utf-8")).get("talker_config", {}).get("spk_id") or {}


def prepare_gguf(ck: dict, a) -> tuple[Path, Path | None]:
    """(GGUF dir, the scoring copy to delete afterwards or None). The training output is never written to: anything
    that changes weights or config happens in a copy under <out>/exports.

    A step-<N> is scored in its OWN voice when --own-voice names it (a speaker phase: the running mean it saved),
    otherwise in the --bake voice from --voice-source (a language phase has no voice of its own). An export that
    already carries --speaker (a speaker phase's final_merged) renders as it is; one that does not (the untrained
    base, a language phase's final_merged) gets the --bake voice in a copy."""
    import gguf_backend as gb
    exports = a.out / "exports"
    copy, borrowed = None, False
    if ck["kind"] == "step":
        scale = (a.lora_alpha / a.lora_rank) if (a.lora_rank and a.lora_alpha) else None
        own = parse_voices(a.own_voice)
        copy = materialize(ck["path"], a.init, exports / ck["label"], scale, own or None)
        borrowed = not own
    elif a.speaker not in speakers_of(ck["path"]):
        copy = exports / ck["label"]
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(ck["path"], copy)
        borrowed = True
    if borrowed:
        if not a.bake:
            raise SystemExit(f"refusing: {ck['label']} carries no voice {a.speaker!r}; give --bake NAME=ROW and "
                             f"--voice-source (or --own-voice for a speaker phase's steps)")
        import bake_voices
        print(json.dumps(bake_voices.bake(copy, a.voice_source, parse_voices(a.bake)), ensure_ascii=False), flush=True)
    if copy is not None and json.loads((copy / "config.json").read_text(encoding="utf-8")).get(
            "tts_model_type") != "custom_voice":
        print(json.dumps(set_custom_voice(copy), ensure_ascii=False), flush=True)
    key = f"{a.key_prefix}-{ck['label']}"
    return gb.ensure_converted(copy or ck["path"], key=key, quant=a.quant), copy   # cached; re-converted on change


def render_arm(arm: str, gguf_dir: Path, lines: Path, renders: Path, draws: int, speaker: str) -> None:
    import render_short
    render_short.render(arm, gguf_dir, lines, renders, draws, speaker, None)


# ---------------------------------------------------------------------------------------------------- judging, measures
def asr_judge(device: str):
    """The release gate's recogniser (src/steps.py), loaded once: path -> transcript."""
    import steps
    model = steps._asr(device)
    return lambda path: model.transcribe(str(path), language="Cantonese")[0].text


def load_t2s():
    """The release gate's script fold (release_gate.load_t2s): refuses without OpenCC, never falls back."""
    import release_gate
    return release_gate.load_t2s()


def clip_measures(row: dict, hyp: str, rules: dict, t2s) -> dict:
    """CER and skipped text on the release gate's script fold (both sides to simplified, CJK only), so a recogniser
    writing the same words in the other script is not counted as error or lost text. The unfolded CER is kept beside
    it, reported only."""
    import release_gate
    import skipped_text as sk
    from steps import cer
    ref = sk.cjk(row["text"])
    speech = float(row["speech_s"])
    return {"clip_id": row["clip_id"], "cer": round(release_gate.folded_scores(row["text"], hyp or "", t2s)["cer"], 4),
            "cer_unfolded": round(cer(ref, sk.cjk(hyp or "")), 4),
            "runaway": speech / max(1, len(ref)) > rules["runaway_s_per_char"],
            "silent": speech < rules["silence_speech_s"] or float(row.get("speech_rms", 1.0)) < rules["silence_rms"],
            "skipped": bool(sk.lost_runs(t2s.convert(row["text"]), t2s.convert(hyp or ""), rules["skip_min_run"])),
            "backend": row.get("backend")}


def aggregate(clips: list[dict]) -> dict:
    import statistics
    n = len(clips)
    if not n:
        raise SystemExit("refusing: a checkpoint with no scored clips")
    return {"n": n, "cer_mean": round(sum(c["cer"] for c in clips) / n, 4),
            "cer_median": round(statistics.median(c["cer"] for c in clips), 4),
            "runaway_rate": round(sum(c["runaway"] for c in clips) / n, 4),
            "silence_rate": round(sum(c["silent"] for c in clips) / n, 4),
            "skipped_rate": round(sum(c["skipped"] for c in clips) / n, 4),
            "cer_unfolded_mean": round(sum(c["cer_unfolded"] for c in clips) / n, 4)}


def one_backend(rows: list[dict]) -> str:
    b = sorted({str(r.get("backend")) for r in rows})
    if len(b) != 1:
        raise SystemExit(f"refusing: the clips came from more than one backend {b}; one comparison, one backend")
    return b[0]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def judged(arm_dir: Path, judge_once) -> list[dict]:
    """Every manifest row with its transcript; transcripts are kept in heard.jsonl so a re-run judges only new clips."""
    rows = read_jsonl(arm_dir / "manifest.jsonl")
    if not rows:
        return []
    heard_p = arm_dir / "heard.jsonl"
    heard = {r["clip_id"]: r["hyp"] for r in read_jsonl(heard_p)}
    with open(heard_p, "a", encoding="utf-8") as fh:
        for r in rows:
            if r["clip_id"] not in heard:
                heard[r["clip_id"]] = judge_once()(arm_dir / r["flac"])
                fh.write(json.dumps({"clip_id": r["clip_id"], "hyp": heard[r["clip_id"]], "judge": JUDGE},
                                    ensure_ascii=False) + "\n")
    return [{**r, "hyp": heard[r["clip_id"]], "flac": str(arm_dir / r["flac"])} for r in rows]


def load_hook(spec: str | None):
    """None when no scorer is set: then nothing is imported."""
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    if not (mod and fn):
        raise SystemExit(f"--scorer {spec!r}: expected module:function")
    return getattr(importlib.import_module(mod), fn)


def floor_lines(lines: Path, out: Path, mode: str) -> Path:
    if mode == "repeat":
        return lines
    p = out / "floor-lines.jsonl"
    rows = read_jsonl(lines)
    p.write_text("".join(json.dumps({**r, "digest": f"{r['digest']}~floor"}, ensure_ascii=False) + "\n"
                         for r in rows), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------------------------------- the run
def finite(v) -> float | None:
    """A number for JSON, or None: summary.json never carries NaN or infinity (json.dumps would write invalid JSON)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def weights_bytes(p: Path) -> int:
    files = sorted(p.glob("model*.safetensors"))
    return sum(f.stat().st_size for f in files)


def disk_report(cks: list[dict]) -> dict:
    """Printed before any work: what each checkpoint holds on disk and the total (keeping every step-<N> costs this)."""
    rows = {c["label"]: weights_bytes(c["path"]) for c in cks}
    total = sum(rows.values())
    for label, b in rows.items():
        print(f"  {label:20s} {b / 1e9:8.2f} GB", flush=True)
    print(f"  {'total':20s} {total / 1e9:8.2f} GB across {len(rows)} checkpoints; each scoring copy under "
          f"<out>/exports is about one checkpoint's size and is removed after conversion unless --keep-exports",
          flush=True)
    return {"bytes": rows, "total_bytes": total}


def score(a, prepare=prepare_gguf, render=render_arm, judge_factory=asr_judge, mlflow_mod=None,
          proc_root: Path = Path("/proc")) -> dict:
    hook = load_hook(a.scorer)                               # fails on a bad spec before any GPU work
    t2s = load_t2s()                                         # refuses without OpenCC, before any GPU work
    guarded = guard(a.beside_training, proc_root)
    cks = [checkpoint_entry(str(p)) for root in a.checkpoints for p in discover(root, required=not a.checkpoint)]
    cks += [checkpoint_entry(s) for s in a.checkpoint]
    if not cks:
        raise SystemExit("refusing: no checkpoint named (--checkpoints DIR or --checkpoint PATH)")
    if any(c["kind"] == "step" for c in cks) and not (a.init and a.init.is_dir()):
        raise SystemExit(f"refusing: step checkpoints need --init, the model the phase started from ({a.init})")
    base = None
    if a.init and not a.no_base:
        base = {"label": "base", "path": a.init, "step": 0, "kind": "export"}
        cks.insert(0, base)
    refuse_bad([c["path"] for c in cks])
    if len({c["label"] for c in cks}) != len(cks):
        raise SystemExit(f"refusing: two checkpoints share a name: {[c['label'] for c in cks]}")
    lines = read_jsonl(a.lines)
    if not lines or len({r["digest"] for r in lines}) != len(lines):
        raise SystemExit(f"refusing: {a.lines} is empty or its digests are not unique")
    rules = {**RULES, **json.loads(a.rules or "{}")}
    scored = [c for c in cks if c is not base] or cks
    floor_ck = next((c for c in cks if c["label"] == a.floor_checkpoint), None) if a.floor_checkpoint else scored[-1]
    if floor_ck is None:
        raise SystemExit(f"refusing: --floor-checkpoint {a.floor_checkpoint} is not among {[c['label'] for c in cks]}")
    print("checkpoints on disk:", flush=True)
    sizes = disk_report(cks)
    a.out.mkdir(parents=True, exist_ok=True)
    renders = a.out / "renders"
    want = len(lines) * a.draws
    flines = floor_lines(a.lines, a.out, a.floor_mode)
    floor_arm = f"{floor_ck['label']}~floor"
    complete = lambda arm: len(read_jsonl(renders / arm / "manifest.jsonl")) >= want

    # Render every checkpoint first, then judge: the renderer and the recogniser are never on the card together.
    for ck in cks:
        todo = [(ck["label"], a.lines)] + ([(floor_arm, flines)] if ck is floor_ck else [])
        todo = [(arm, ln) for arm, ln in todo if not complete(arm)]
        if not todo:
            continue
        gguf, copy = prepare(ck, a)
        for arm, ln in todo:
            render(arm, gguf, ln, renders, a.draws, a.speaker)
        if copy is not None and not a.keep_exports:
            shutil.rmtree(copy, ignore_errors=True)

    judge = {}

    def judge_once():
        if "j" not in judge:
            judge["j"] = judge_factory(a.asr_device)
        return judge["j"]
    clip_rows, readings = {}, []
    for arm in [c["label"] for c in cks] + [floor_arm]:
        rows = judged(renders / arm, judge_once)
        if len(rows) < want:
            raise SystemExit(f"refusing: {arm} has {len(rows)} of {want} clips rendered")
        clip_rows[arm] = rows
    backend = one_backend([r for rows in clip_rows.values() for r in rows])
    measured = {arm: [clip_measures(r, r["hyp"], rules, t2s) for r in rows] for arm, rows in clip_rows.items()}
    agg = {arm: aggregate(m) for arm, m in measured.items()}
    hooks, nulls = {}, []
    if hook:
        for ck in cks + [{"label": floor_arm, "step": floor_ck["step"]}]:
            out = {}
            for k, v in (hook(label=ck["label"], step=ck["step"], clips=clip_rows[ck["label"]]) or {}).items():
                out[k] = finite(v)
                if out[k] is None:
                    nulls.append(f"hook/{k} on {ck['label']}: the scorer returned {v!r}, not a finite number")
            hooks[ck["label"]] = out
    floor = {m: round(abs(agg[floor_ck["label"]][m] - agg[floor_arm][m]), 4) for m in MEASURES}
    hook_floor = {}
    for k, v in (hooks.get(floor_ck["label"], {}) if hook else {}).items():
        w = hooks[floor_arm].get(k)
        hook_floor[k] = round(abs(v - w), 4) if v is not None and w is not None else None
        if hook_floor[k] is None:
            nulls.append(f"hook/{k} floor: missing or not finite on one of the two scorings of {floor_ck['label']}")
    for ck in cks:
        A = agg[ck["label"]]
        readings.append({
            "label": ck["label"], "step": ck["step"], "path": str(ck["path"]), "backend": backend, "n": A["n"],
            "measures": {m: {"value": A[m], "floor": floor[m],
                             "vs_base": round(A[m] - agg["base"][m], 4) if base else None} for m in MEASURES},
            **({"hook": {k: {"value": v, "floor": hook_floor.get(k),
                             "vs_base": (round(v - hooks["base"][k], 4) if base and v is not None
                                         and hooks["base"].get(k) is not None else None)}
                         for k, v in hooks[ck["label"]].items()}} if hook else {})})
    summary = {
        "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checkpoints": readings,
        "base": ({"label": "base", "path": str(a.init), "note": "the untrained --init on the same lines, seeds, voice "
                  "and judge; every reading's vs_base is its value minus this row's",
                  **({"voice_caveat": "speaker phase: the base row speaks the borrowed --bake voice and each step its "
                      "own trained voice (--own-voice), so vs_base measures the voice change as well as the training"}
                     if a.own_voice else {})} if base else
                 {"label": None, "reason": "--no-base" if a.no_base else "no --init given"}),
        "floor": {"checkpoint": floor_ck["label"], "mode": a.floor_mode, "first": agg[floor_ck["label"]],
                  "second": agg[floor_arm], "difference": floor, **({"hook": hook_floor} if hook else {}),
                  "note": "one sample: |first - second| over one pair of scorings, not a confidence bound"},
        "rules": rules, "rules_status": RULES_STATUS, "cer": "script-folded (release_gate.folded_scores); "
        "cer_unfolded_mean is reported only", "lines": str(a.lines),
        "lines_sha256": hashlib.sha256(a.lines.read_bytes()).hexdigest(),
        "draws": a.draws, "speaker": a.speaker, "bake": a.bake, "own_voice": a.own_voice, "backend": backend,
        "judge": JUDGE, "seed_rule": "render_short.SEED_RULE: int(sha256(f'{digest}:{draw}')[:8], 16) per clip",
        "scorer": a.scorer or None, "guard": {**guarded, "sees": "trainer processes on this host only"},
        "disk": sizes, "nulls": nulls,
        "note": "reports; never chooses a checkpoint. A rate may veto on a functional failure; the floor is the "
                "difference two scorings of one checkpoint give, and a difference inside it means nothing."}
    if not a.no_mlflow:
        summary["mlflow_run_id"] = log_mlflow(summary, a, mlflow_mod)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                        encoding="utf-8")
    with open(a.out / "clips.jsonl", "w", encoding="utf-8") as fh:
        for arm, m in measured.items():
            for c in m:
                fh.write(json.dumps({"arm": arm, **c}, ensure_ascii=False) + "\n")
    print(table(summary))
    return summary


def log_mlflow(summary: dict, a, mlflow_mod=None) -> str:
    """One MLflow run per scoring: each measure at step N with its floor beside it as <measure>_floor. A checkpoint
    with no step number (an export named without @STEP) logs under <label>/<measure>."""
    mlflow = mlflow_mod
    if mlflow is None:
        import mlflow
    mlflow.set_tracking_uri(a.mlflow_uri)
    mlflow.set_experiment(a.mlflow_experiment)
    with mlflow.start_run(run_name=f"score-{a.out.name}") as run:
        mlflow.log_params({"lines_sha256": summary["lines_sha256"], "draws": a.draws, "speaker": a.speaker,
                           "backend": summary["backend"], "floor_mode": a.floor_mode,
                           "floor_checkpoint": summary["floor"]["checkpoint"], "scorer": a.scorer or "none",
                           "rules": json.dumps(summary["rules"])})
        for r in summary["checkpoints"]:
            pre, step = ("", r["step"]) if r["step"] is not None else (f"{r['label']}/", 0)
            vals = {**{m: v for m, v in r["measures"].items()},
                    **{f"hook/{k}": v for k, v in r.get("hook", {}).items()}}
            for m, v in vals.items():
                for suffix, x in (("", v["value"]), ("_floor", v["floor"]), ("_vs_base", v.get("vs_base"))):
                    if x is not None:
                        mlflow.log_metric(f"{pre}{m}{suffix}", x, step=step)
        mlflow.log_dict(summary, "checkpoint_scores.json")
        return run.info.run_id


def table(s: dict) -> str:
    L = [f"{'checkpoint':16s} " + " ".join(f"{m:>20s}" for m in MEASURES)]
    for r in s["checkpoints"]:
        L.append(f"{r['label']:16s} " + " ".join(f"{r['measures'][m]['value']:>9.4f} (+-{r['measures'][m]['floor']:.4f})"
                                                  for m in MEASURES))
    return "\n".join(L) + f"\n(+-: the no-op floor, {s['floor']['mode']} on {s['floor']['checkpoint']})"


# ---------------------------------------------------------------------------------------------------- self-test, CLI
def self_test() -> int:
    """Each clip rule fires on a planted defect and reads nothing on a faithful clip."""
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
        fails += not cond
    t2s = load_t2s()
    text = "今日天氣好好所以我哋一齊去公園"                      # invented for this test, 15 characters
    ok = {"clip_id": "a", "text": text, "speech_s": 3.6, "speech_rms": 0.05, "backend": "gguf"}
    m = clip_measures(ok, text, RULES, t2s)
    check("a faithful clip: CER 0, no rule fires", m["cer"] == 0 and not (m["runaway"] or m["silent"] or m["skipped"]), str(m))
    check("RED: speech three times as long per character is a runaway",
          clip_measures({**ok, "speech_s": 3.6 * 3}, text, RULES, t2s)["runaway"])
    check("RED: a clip of 0.1 s speech is silence", clip_measures({**ok, "speech_s": 0.1}, text, RULES, t2s)["silent"])
    check("RED: a near-zero RMS is silence", clip_measures({**ok, "speech_rms": 0.0001}, text, RULES, t2s)["silent"])
    check("RED: six characters unheard is skipped text", clip_measures(ok, text[:4] + text[10:], RULES, t2s)["skipped"])
    check("one character misheard is not skipped text", not clip_measures(ok, text[:3] + "晏" + text[4:], RULES, t2s)["skipped"])
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--checkpoints", type=Path, action="append", default=[],
                    help="a directory of step-<N> checkpoints (the phase's resume/); every one is scored")
    ap.add_argument("--checkpoint", action="append", default=[], help="PATH[@STEP]: one more checkpoint (an export "
                    "such as final_merged, or one step-<N>)")
    ap.add_argument("--init", type=Path, help="the model the phase started from (its files and key set)")
    ap.add_argument("--lines", type=Path, help='JSONL of {"digest", "text"}: the fixed lines')
    ap.add_argument("--out", type=Path)
    ap.add_argument("--speaker", help="the voice every checkpoint renders with")
    ap.add_argument("--bake", action="append", default=[], help="NAME=ROW baked from --voice-source into the scoring copy")
    ap.add_argument("--voice-source", type=Path)
    ap.add_argument("--own-voice", action="append", default=[], help="NAME=ROW: bake each step's OWN running mean "
                    "(voice_means.safetensors) at ROW, as the speaker phase's export does")
    ap.add_argument("--no-base", action="store_true", help="do not score the untrained --init as the step-0 row")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--lora-rank", type=float)
    ap.add_argument("--lora-alpha", type=float)
    ap.add_argument("--quant", default="q8_0")
    ap.add_argument("--key-prefix", help="GGUF cache key prefix (default score-<out dir name>)")
    ap.add_argument("--floor-checkpoint", help="the checkpoint scored twice (default: the last)")
    ap.add_argument("--floor-mode", choices=("reseed", "repeat"), default="reseed")
    ap.add_argument("--rules", help="JSON overriding RULES")
    ap.add_argument("--scorer", help="module:function, an extra scorer; default none (nothing imported)")
    ap.add_argument("--beside-training", action="store_true", help="score although a trainer is running")
    ap.add_argument("--keep-exports", action="store_true", help="keep the materialized scoring copies")
    ap.add_argument("--asr-device", default="cuda:0")
    ap.add_argument("--mlflow-uri", help="tracking URI (default sqlite:///<out>/mlflow.db)")
    ap.add_argument("--mlflow-experiment", default="checkpoint-scores")
    ap.add_argument("--no-mlflow", action="store_true")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.lines and a.out and a.speaker):
        parser().error("--lines, --out and --speaker are required")
    if a.bake and not a.voice_source:
        parser().error("--bake needs --voice-source")
    a.out = a.out.expanduser()
    a.key_prefix = a.key_prefix or f"score-{a.out.name}"
    a.mlflow_uri = a.mlflow_uri or f"sqlite:///{a.out / 'mlflow.db'}"
    a.out.mkdir(parents=True, exist_ok=True)
    score(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
