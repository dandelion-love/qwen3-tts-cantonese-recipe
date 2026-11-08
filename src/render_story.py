#!/usr/bin/env python3
"""Read long prose aloud on one or more GGUF voices: one continuous read per text per arm, for listening and demos.

Each text file holds a title line, a blank line, and paragraphs. A paragraph is chunked at clause punctuation into
the chunk ceiling --hi (default 45; the lab's chunk-ceiling concept) (chunking.chunk, which keeps each chunk's own closing
punctuation), every chunk is rendered on every arm at the SAME seed (gguf, seeded talker sampling, sub-talker
sampling on), edge silence is trimmed to EDGE_S, and the read is joined with a pause chosen by the punctuation that
ended the chunk (longer after a sentence, longer still after a paragraph) and level-matched across arms.

BY DEFAULT EVERY CHUNK IS THE MODEL'S FIRST DRAW, defects and all, so a listening test hears what the model does
(owner 2026-10-19: hiding the runaways "is a pretty big miss for testing the model. We should keep those and have
them be listenable"). Only a published demo, a consumption artefact, may pass --redraw N: a chunk whose render is a
functional failure is then re-drawn at a new seed, at most N times -- a runaway (speech over RUNAWAY_SPC seconds per
character) or a near-silence (under SILENT_SPC). Either way each chunk's first-draw defect and rate are reported in
the manifest, and nothing else is cleaned. Text is never rewritten: English, numerals or anything the model reads
badly stays as written.

    python render_story.py --texts DIR --out DIR --arm NAME=GGUF_DIR:SPEAKER [--arm ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/: gguf_backend, audio_trim, audio_uri, chunking

EDGE_S = 0.12
PAUSE = {"，": 0.20, "、": 0.16, "：": 0.30, "；": 0.32, "。": 0.48, "！": 0.48, "？": 0.50, "": 0.18}
PARA_EXTRA = 0.45
RUNAWAY_SPC, SILENT_SPC, DEMO_REDRAW = 0.60, 0.07, 4
TARGET_RMS_DB = -20.0


# WHAT MADE A READ. A read rendered through qwen3-tts.cpp can be reproduced only with its seeds, the upstream commit the
# engine was built at, the patches applied to it, and the codec it decoded with, so the manifest records all of them.
SEED_RULE = {"function": "seed_of", "key": "story|chunk|text|draw, then |salt when a salt is set",
             "hash": "the first 8 hex digits of the key's sha256, as an integer", "per": "request"}
RECIPE_ROOT = Path(__file__).resolve().parents[1]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def engine_block() -> dict:
    """The engine this run renders with. Every story render goes through the GGUF backend. A fact that cannot be
    read here (no source checkout, no codec) is None, never guessed."""
    import gguf_backend as gb
    pin_expected = None
    try:
        m = re.search(r"QWEN_TTS_CPP_PIN:-([0-9a-f]{40})",
                      (RECIPE_ROOT / "scripts" / "build_gguf_runtime.sh").read_text(encoding="utf-8"))
        pin_expected = m.group(1) if m else None
    except OSError:
        pass
    patches = RECIPE_ROOT / "external" / "qwen3-tts-cpp" / "patches"
    expected = {p.name: _sha256(p) for p in sorted(patches.glob("*.patch"))}
    cli = gb.CLI_BIN
    cli_hash = _sha256(cli) if cli.is_file() else None
    stamp = None
    try:
        candidate = json.loads(Path(str(cli) + ".build.json").read_text(encoding="utf-8"))
        if (isinstance(candidate, dict) and candidate.get("version") == 1 and cli_hash
                and candidate.get("cli_sha256") == cli_hash):
            stamp = candidate
    except (OSError, ValueError):
        pass
    codec = gb._find_codec()
    return {"backend": "gguf", "engine": "qwen3-tts.cpp", "upstream": "https://github.com/Danmoreng/qwen3-tts.cpp",
            "pin_expected": pin_expected, "pin_built": stamp.get("pin_built") if stamp else None,
            "ggml_commit": stamp.get("ggml_commit") if stamp else None,
            "patches_expected": expected, "patches_built": stamp.get("patches_built") if stamp else None,
            "cli_sha256": cli_hash, "codec": codec.name if codec else None,
            "codec_sha256": _sha256(codec) if codec else None}



def provenance_problems(man: dict) -> list[str]:
    """Every reason a story manifest cannot say what made its read; empty when it can."""
    eng = man.get("engine")
    if not isinstance(eng, dict):
        return ["no engine block: the backend, the upstream pin and patches, and the codec are unknown"]
    why = []
    backend = eng.get("backend")
    if backend not in ("gguf", "torch"):
        why.append(f"engine.backend is {backend!r}, not gguf or torch")
    if backend == "gguf":
        for field, size in (("pin_expected", 40), ("pin_built", 40), ("ggml_commit", 40),
                            ("cli_sha256", 64), ("codec_sha256", 64)):
            value = eng.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % size, value):
                why.append("engine." + field + " is missing or invalid")
        if eng.get("pin_built") and eng.get("pin_expected") and eng["pin_built"] != eng["pin_expected"]:
            why.append("engine.pin_built differs from engine.pin_expected")
        for field in ("patches_expected", "patches_built"):
            value = eng.get(field)
            if not isinstance(value, dict) or not all(isinstance(k, str) and k and isinstance(v, str)
                                                       and re.fullmatch(r"[0-9a-f]{64}", v)
                                                       for k, v in value.items()):
                why.append("engine." + field + " is missing or invalid")
        if isinstance(eng.get("patches_expected"), dict) and isinstance(eng.get("patches_built"), dict) and eng["patches_expected"] != eng["patches_built"]:
            why.append("engine.patches_built differs from engine.patches_expected")
        if not isinstance(eng.get("codec"), str) or not eng["codec"]:
            why.append("engine.codec is missing")
    if not isinstance((man.get("rule") or {}).get("seed_rule"), dict):
        why.append("rule.seed_rule is missing")
    return why


def require_provenance(man: dict, where) -> None:
    """Refuse a manifest that cannot say what made its read, naming every missing field."""
    why = provenance_problems(man)
    if why:
        raise SystemExit(f"{where}: the story manifest cannot say what made this read: {'; '.join(why)}. "
                         "Re-render it with src/render_story.py.")


def read_text(p: Path) -> tuple[str, list[str]]:
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    title = lines[0].strip()
    paras = [x.strip() for x in "\n".join(lines[1:]).split("\n") if x.strip()]
    return title, paras


def plan(paras: list[str], hi: int = 26) -> list[dict]:
    from chunking import chunk
    out = []
    for pi, para in enumerate(paras):
        cs = chunk(para, 14, hi)
        for ci, (txt, punct) in enumerate(cs):
            out.append({"para": pi, "text": txt + punct, "punct": punct, "last_in_para": ci == len(cs) - 1})
    return out


def seed_of(story: str, i: int, text: str, k: int = 0, salt: str = "") -> int:
    """Each chunk's seed. `salt` (default empty: the seeds every earlier read used) gives the same text a fresh,
    still reproducible, draw -- for measuring how much two draws of one model differ."""
    key = f"{story}|{i}|{text}|{k}" + (f"|{salt}" if salt else "")
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def cjk_len(s: str) -> int:
    return sum(1 for ch in s if "㐀" <= ch <= "鿿" or "\U00020000" <= ch <= "\U0002ffff")


def judge(y, sr, text: str) -> tuple[str | None, float]:
    from audio_trim import trim_edges
    _, edge = trim_edges(y, sr, max_edge_s=0.0)
    speech = (edge["end_sample"] - edge["start_sample"]) / sr
    n = max(1, cjk_len(text) + sum(1 for ch in text if ch.isascii() and ch.isalnum()) // 3)
    spc = speech / n
    return ("runaway" if spc > RUNAWAY_SPC else "silent" if spc < SILENT_SPC else None), round(spc, 3)


def punctuation_pause(punct: str, pauses: dict, fallback: float) -> float:
    """Use the first recognized mark, including when quotes or marks follow it."""
    return next((pauses[ch] for ch in punct if ch in pauses), fallback)


def render_arm(name: str, gguf: Path, speaker: str, stories: dict, out: Path, redraw: int = 0,
               salt: str = "") -> dict:
    import gguf_backend as gb
    from audio_trim import trim_edges
    m = gb.GGUFModel(gguf, gguf.name, max_edge_s=None)
    rec = {}
    try:
        for sid, (title, chunks) in stories.items():
            clips, rows = [], []
            for i, c in enumerate(chunks):
                first = None
                for k in range(redraw + 1):
                    seed = seed_of(sid, i, c["text"], k, salt)
                    if speaker.startswith("instruct:"):     # a VoiceDesign model: the voice is a description
                        ws, sr = m.generate_voice_design(text=[c["text"]], instruct=speaker[len("instruct:"):],
                                                         language="chinese", do_sample=False, seed=seed)
                    else:
                        ws, sr = m.generate_custom_voice(text=[c["text"]], speaker=speaker, language="chinese",
                                                         do_sample=False, seed=seed)
                    y = np.asarray(ws[0], dtype="float32")
                    bad, spc = judge(y, sr, c["text"])
                    if k == 0:
                        first, first_spc = bad, spc
                    if bad is None:
                        break
                y, _ = trim_edges(y, sr, max_edge_s=EDGE_S)
                clips.append((np.asarray(y, dtype="float32"), sr, c))
                rows.append({"i": i, "text": c["text"], "seed": seed, "redraws": k, "first_draw_defect": first,
                             "final_defect": bad, "s_per_char": spc,
                             "first_s_per_char": first_spc})
            sr = clips[0][1]
            parts, at = [], 0
            for (y, _, c), row in zip(clips, rows):
                row["span"] = [at, at + len(y)]            # samples in the read, for cutting it by paragraph later
                parts.append(y)
                at += len(y) + int((punctuation_pause(c["punct"], PAUSE, PAUSE[""]) + (PARA_EXTRA if c["last_in_para"] else 0.0)) * sr)
                pause = punctuation_pause(c["punct"], PAUSE, PAUSE[""]) + (PARA_EXTRA if c["last_in_para"] else 0.0)
                parts.append(np.zeros(int(pause * sr), dtype="float32"))
            read = np.concatenate(parts[:-1])
            loud = read[np.abs(read) > 1e-3]
            g = 10 ** ((TARGET_RMS_DB - 20 * np.log10(float(np.sqrt(np.mean(loud ** 2))) + 1e-9)) / 20) if len(loud) else 1.0
            read = np.clip(read * g, -0.999, 0.999).astype("float32")
            rec[sid] = {"title": title, "chunks": rows, "seconds": round(len(read) / sr, 2), "sr": sr,
                        "first_draw_defects": sum(r["first_draw_defect"] is not None for r in rows),
                        "final_defects": sum(r["final_defect"] is not None for r in rows), "read": read}
            print(f"[{time.strftime('%H:%M:%S')}] {name} {sid}: {len(rows)} chunks, {rec[sid]['seconds']} s, "
                  f"first-draw defects {rec[sid]['first_draw_defects']}, left {rec[sid]['final_defects']}", flush=True)
    finally:
        m.close()
    return rec


def main() -> int:
    from audio_uri import write_flac
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=GGUF_DIR:SPEAKER, or NAME=GGUF_DIR:instruct:DESCRIPTION for a VoiceDesign model")
    ap.add_argument("--redraw", type=int, default=0, help="re-draw a runaway or silent chunk up to N times "
                    f"(published demos only, e.g. {DEMO_REDRAW}); default 0 keeps every first draw, for listening tests")
    ap.add_argument("--seed-salt", default="", help="a fresh, reproducible draw of every chunk (default: the "
                    "seeds every earlier read used)")
    # 45 since 2026-10-19 (it was 70 for a few hours, 26 before): longer chunks read better (the owner, BM-DEMO-091 at
    # 26: it "makes the model sound worse than it is suppose to be"), but at 70 the model SKIPS whole clauses from
    # 50-63-character chunks (EXP-090 by skipped_text.py: 3-5 chunks with a lost run against none at 26 or 45). The
    # owner: "The quality is better with higher characters but dropped characters are no good." EXP-094 settles it.
    ap.add_argument("--hi", type=int, default=45, help="the chunk ceiling in characters (synthesize_long.chunk's hi; 26 was the earlier band, still in the v0.0.4 model card's demo pages)")
    a = ap.parse_args()
    stories = {}
    for p in sorted(a.texts.glob("*.txt")):
        title, paras = read_text(p)
        stories[p.stem] = (title, plan(paras, a.hi))
    a.out.mkdir(parents=True, exist_ok=True)
    man = {"texts": {sid: {"title": t, "sha256": hashlib.sha256((a.texts / f"{sid}.txt").read_bytes()).hexdigest(),
                           "chunks": len(c)} for sid, (t, c) in stories.items()},
           "rule": {"chunk_hi": a.hi, "seed_salt": a.seed_salt, "edge_s": EDGE_S, "pause": PAUSE, "para_extra": PARA_EXTRA, "runaway_spc": RUNAWAY_SPC,
                    "silent_spc": SILENT_SPC, "max_redraw": a.redraw, "target_rms_db": TARGET_RMS_DB,
                    "seed_rule": SEED_RULE},
           "engine": engine_block(),
           "arms": {}}
    require_provenance(man, "runtime")
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        gguf, speaker = rest.split(":instruct:", 1) if ":instruct:" in rest else rest.rsplit(":", 1)
        speaker = ("instruct:" + speaker) if ":instruct:" in rest else speaker
        gguf = Path(gguf).expanduser()
        rec = render_arm(name, gguf, speaker, stories, a.out, a.redraw, a.seed_salt)
        arm = {"gguf": str(gguf), "speaker": speaker, "stories": {}}
        for sid, r in rec.items():
            f = a.out / name / f"{sid}.flac"
            f.parent.mkdir(parents=True, exist_ok=True)
            write_flac(f, r.pop("read"), r["sr"])
            arm["stories"][sid] = {**r, "flac": str(f.relative_to(a.out))}
        man["arms"][name] = arm
    (a.out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
