#!/usr/bin/env python3
"""Bake the chosen voice vectors into a merged speaker-phase checkpoint: rows of talker.model.codec_embedding copied
from a source checkpoint, and the config's spk_id naming them.

WHY A BAKE AND NOT THE TRAINED ROW. The speaker phase writes the voice it trained (the mean speaker-encoder vector of
its clips) at row 3000. The recipe ships a CHOSEN single-clip vector instead (owner, 2026-10-17: "choose, never
assume"), today v0.0.1's 343 vector at row 3000, and a second voice row costs nothing (the 777 vector at row 3001).
Both come from a checkpoint that already carries them: in the lab, the v0.0.4 release's merged weights
(exp083.stage_bake). This script is that stage with the source as an argument.

What it does, in the target directory, in place:
  * each named row of talker.model.codec_embedding.weight set to the source's same row, cast to the target's dtype;
  * talker_config.spk_id set to exactly the named voices (and spk_is_dialect to False for each, if the key exists);
  * refuses unless every baked row then equals what was meant for it bit for bit, and unless tts_model_size is
    --model-size (1b7 by default). A config with no tts_model_size is refused too: Qwen-TTS Studio's converter
    defaults a missing size to 0b6, a silently mislabelled model.

A SOURCE OF ANOTHER WIDTH IS NEVER COPIED. The codec embedding is as wide as the talker: 2048 on the 1.7B base, 1024
on the 0.6B. A 1.7B voice row cannot go into a 0.6B talker, and no slice or projection of it is the same voice. So when
the source's width differs from the target's (or there is no source), each voice is DERIVED instead, by the target's
own speaker encoder, the way the speaker phase writes row 3000: the encoder run over the voice's clips (--voice-audio,
once per clip; 128-bin mel exactly as trainer/dataset.py's extract_mels, which the self-test compares against) and the
per-clip vectors averaged in fp32. ONE DIFFERENCE FROM THE TRAINER: it asserts 24 kHz input and refuses anything else,
while this resamples any clip to 24 kHz (librosa) first, so a clip the trainer would refuse still bakes here. On a
24 kHz clip the two give the same mel. When the source IS the target's width, the source row is copied and any clips
given are ignored, with a warning on stderr and `clips_ignored` in the record. The encoder
comes from --encoder, a checkpoint of the target's size that still carries speaker_encoder.* (the recipe's language
export keeps it; the speaker export drops it; no phase trains it, so the public base's encoder is the same weights).
A voice with neither a same-width source row nor clips is refused, and so is an encoder whose width is not the
target's.

It records, as JSON on stdout and in --record: how each row was made (copied or derived, with its clips), each row's
equality with what was meant for it, and the cosine between the trained row 3000 and the baked one (how far the chosen
voice is from the voice the phase learned).

    python src/bake_voices.py --target RUN/spk/final_merged --source RELEASE_MERGED \\
        --voice hk_voice_343=3000 --voice hk_voice_777=3001 --record RUN/bake.json
    python src/bake_voices.py --target RUN/spk/final_merged --model-size 0b6 --encoder RUN/lang/final_merged \\
        --voice-audio hk_voice_343=CLIP1.flac --voice-audio hk_voice_343=CLIP2.flac --voice hk_voice_343=3000
    python src/bake_voices.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEY = "talker.model.codec_embedding.weight"


def codec_file(d: Path) -> Path:
    """The safetensors shard of `d` holding the codec embedding."""
    from safetensors import safe_open
    for f in sorted(Path(d).glob("model*.safetensors")):
        with safe_open(str(f), "pt") as s:
            if KEY in s.keys():
                return f
    raise SystemExit(f"no {KEY} in {d}")


ENC_PREFIX = "speaker_encoder."
SR = 24000
MODEL_SIZES = ("1b7", "0b6")


def require_size(cfg: dict, want: str, where) -> None:
    got = cfg.get("tts_model_size")
    if got is None:
        raise SystemExit(f"{where}: config.json has no tts_model_size (the converter would assume '0b6')")
    if got != want:
        raise SystemExit(f"{where}: tts_model_size is {got!r}, not {want!r}")


def load_clip(path):
    """One clip as float32 mono at 24 kHz (the speaker encoder's rate; the trainer asserts it)."""
    import librosa
    import numpy as np
    wav, _ = librosa.load(str(path), sr=SR, mono=True)
    return wav.astype(np.float32)


def clip_mel(path):
    """The (1, frames, 128) mel the speaker encoder reads, with trainer/dataset.py extract_mels's arguments."""
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    return mel_spectrogram(torch.from_numpy(load_clip(path)).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=SR,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)


def encoder_mean(encoder: Path, clips: list):
    """The mean speaker-encoder vector over `clips`, by the speaker_encoder.* weights in `encoder`: the arithmetic of
    the speaker phase's row 3000 (mel as trainer/dataset.py, encoder in the checkpoint's dtype, fp32 mean)."""
    import torch
    from safetensors import safe_open
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
    cfg = json.loads((Path(encoder) / "config.json").read_text(encoding="utf-8"))
    if "speaker_encoder_config" not in cfg:
        raise SystemExit(f"{encoder}: no speaker_encoder_config; use a checkpoint that keeps the speaker encoder")
    state = {}
    for f in sorted(Path(encoder).glob("model*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k.startswith(ENC_PREFIX):
                    state[k[len(ENC_PREFIX):]] = s.get_tensor(k)
    if not state:
        raise SystemExit(f"{encoder}: no {ENC_PREFIX}* weights (the speaker phase drops them; use the language export "
                         "or the base)")
    dtype = next(iter(state.values())).dtype
    enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**cfg["speaker_encoder_config"]))
    enc.load_state_dict(state, strict=True)
    enc = enc.to(dtype).eval()
    vecs = []
    with torch.inference_mode():
        for c in clips:
            vecs.append(enc(clip_mel(c).to(dtype))[0].float())
    return torch.stack(vecs).mean(0)


def bake(target: Path, source: Path | None, voices: dict[str, int], model_size: str = "1b7",
         encoder: Path | None = None, audio: dict[str, list] | None = None, derive=encoder_mean) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    audio = audio or {}
    cfg_p = target / "config.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    require_size(cfg, model_size, target)
    src = None
    if source is not None:
        with safe_open(str(codec_file(source)), "pt") as s:
            src = s.get_tensor(KEY)
    f = codec_file(target)
    with safe_open(str(f), "pt") as s:
        meta = s.metadata()
    t = load_file(str(f))
    rows, width = t[KEY].shape
    if any(r >= rows or (src is not None and r >= src.shape[0]) for r in voices.values()):
        raise SystemExit(f"a row in {voices} is past the table ({rows} / {None if src is None else src.shape[0]} rows)")
    same_width = src is not None and src.shape[1] == width
    want, how = {}, {}
    for name, row in voices.items():
        if same_width:
            want[name], how[name] = src[row].to(t[KEY].dtype), {"from": "source", "row": row}
            if audio.get(name):
                how[name]["clips_ignored"] = [str(c) for c in audio[name]]
                print(f"WARNING {name}: the source is the target's width ({width}), so its row is copied and the "
                      f"{len(audio[name])} clip(s) given are ignored", file=sys.stderr)
        elif name in audio:
            if encoder is None:
                raise SystemExit(f"{name}: deriving a row needs --encoder (a {model_size} checkpoint with its speaker "
                                 "encoder)")
            v = derive(encoder, audio[name])
            if tuple(v.shape) != (width,):
                raise SystemExit(f"{name}: the encoder gives a {tuple(v.shape)} vector and the target's rows are {width} "
                                 f"wide; --encoder must be the target's size ({model_size})")
            want[name] = v.to(t[KEY].dtype)
            how[name] = {"from": "own-encoder mean", "encoder": str(encoder), "clips": [str(c) for c in audio[name]]}
        else:
            why = ("there is no --source" if src is None else
                   f"the source's rows are {src.shape[1]} wide and the target's {width}, so copying is refused")
            raise SystemExit(f"{name}: {why}; give its clips with --voice-audio {name}=CLIP (once per clip) to derive the row "
                             "by the target's own speaker encoder")
    trained = t[KEY][3000].clone()
    for name, row in voices.items():
        t[KEY][row] = want[name]
    save_file(t, str(f), metadata=meta)
    cfg["talker_config"]["spk_id"] = dict(voices)
    if "spk_is_dialect" in cfg["talker_config"]:
        cfg["talker_config"]["spk_is_dialect"] = {v: False for v in voices}
    cfg_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with safe_open(str(f), "pt") as s:
        back = s.get_tensor(KEY)
    rec = {"target": str(target), "source": None if source is None else str(source), "model_size": model_size,
           "width": width, "spk_id": cfg["talker_config"]["spk_id"], "rows": how,
           "rows_equal_source": {v: bool(torch.equal(back[r], want[v])) for v, r in voices.items()},
           "row3000_cos_trained_vs_baked": round(float(torch.nn.functional.cosine_similarity(
               trained.float(), back[3000].float(), dim=0)), 6)}
    if not all(rec["rows_equal_source"].values()):
        raise SystemExit(f"the baked rows are not the source's: {rec}")
    return rec


def parse_voices(specs: list[str]) -> dict[str, int]:
    out = {}
    for s in specs:
        name, row = s.split("=", 1)
        if name in out or int(row) in out.values():
            raise SystemExit(f"voice {s} repeats a name or a row")
        out[name] = int(row)
    return out


def self_test() -> int:
    """A two-shard fixture: the bake copies exactly the named rows, sets spk_id, and refuses a config of another size
    or none. Then a 1024-wide target (the 0.6B talker's width) against a 2048-wide source: the copy is refused, and
    with clips the row is the target encoder's mean (a small real Qwen3-TTS speaker encoder, random weights)."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        fails += not cond
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "src", Path(td) / "dst"
        for d, fill in ((src, 1.0), (dst, 2.0)):
            d.mkdir()
            save_file({"other.weight": torch.zeros(2)}, str(d / "model-00001-of-00002.safetensors"))
            e = torch.full((3072, 8), fill, dtype=torch.bfloat16)
            e[3000] = torch.arange(8) + fill
            save_file({KEY: e}, str(d / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
            (d / "config.json").write_text(json.dumps({"tts_model_size": "1b7", "talker_config": {
                "spk_id": {"hk_voice_343": 3000}, "spk_is_dialect": {"hk_voice_343": False}}}), encoding="utf-8")
        rec = bake(dst, src, {"hk_voice_343": 3000, "hk_voice_777": 3001})
        from safetensors.torch import load_file
        t = load_file(str(dst / "model-00002-of-00002.safetensors"))[KEY]
        check("the named rows equal the source's", all(rec["rows_equal_source"].values())
              and torch.equal(t[3001], torch.full((8,), 1.0, dtype=torch.bfloat16)))
        check("RED: an unnamed row is untouched", torch.equal(t[2999], torch.full((8,), 2.0, dtype=torch.bfloat16)))
        cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))["talker_config"]
        check("spk_id names exactly the baked voices", cfg["spk_id"] == {"hk_voice_343": 3000, "hk_voice_777": 3001}
              and cfg["spk_is_dialect"] == {"hk_voice_343": False, "hk_voice_777": False})
        check("the trained-vs-baked cosine is recorded", -1.0 <= rec["row3000_cos_trained_vs_baked"] <= 1.0)
        bad = json.loads((dst / "config.json").read_text(encoding="utf-8"))
        bad["tts_model_size"] = "0b6"
        (dst / "config.json").write_text(json.dumps(bad), encoding="utf-8")
        try:
            bake(dst, src, {"hk_voice_343": 3000})
            check("RED: a 0b6 config is refused", False)
        except SystemExit:
            check("RED: a 0b6 config is refused", True)
        try:
            bake(dst, src, {"hk_voice_343": 3000}, model_size="0b6")
            check("a 0b6 config is accepted when --model-size is 0b6", True)
        except SystemExit as e:
            check(f"a 0b6 config is accepted when --model-size is 0b6 ({e})", False)
        del bad["tts_model_size"]
        (dst / "config.json").write_text(json.dumps(bad), encoding="utf-8")
        for size in MODEL_SIZES:
            try:
                bake(dst, src, {"hk_voice_343": 3000}, model_size=size)
                check(f"RED: a config with no tts_model_size is refused (--model-size {size})", False)
            except SystemExit:
                check(f"RED: a config with no tts_model_size is refused (--model-size {size})", True)
    width_mismatch_test(check)            # counts its failures through check; its return value is not a count
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


def width_mismatch_test(check) -> None:
    """A 0b6 target 1024 wide, a source 2048 wide. Failures are counted by `check`."""
    import contextlib
    import io
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file, save_file
    W, SRC_W = 1024, 2048
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, dst, enc_dir = td / "src", td / "dst", td / "enc"
        for d in (src, dst, enc_dir):
            d.mkdir()
        s = torch.full((3072, SRC_W), 1.0, dtype=torch.bfloat16)
        save_file({KEY: s}, str(src / "model.safetensors"), metadata={"format": "pt"})
        (src / "config.json").write_text(json.dumps({"tts_model_size": "1b7", "talker_config": {}}), encoding="utf-8")
        e = torch.full((3072, W), 2.0, dtype=torch.bfloat16)
        save_file({KEY: e}, str(dst / "model.safetensors"), metadata={"format": "pt"})
        (dst / "config.json").write_text(json.dumps({"tts_model_size": "0b6", "talker_config": {
            "spk_id": {"hk_voice_343": 3000}}}), encoding="utf-8")
        before = (dst / "model.safetensors").read_bytes()
        try:
            bake(dst, src, {"hk_voice_343": 3000}, model_size="0b6")
            check("RED: a 2048-wide source row is not copied into a 1024-wide target", False)
        except SystemExit as ex:
            check("RED: a 2048-wide source row is not copied into a 1024-wide target (refused: "
                  f"{str(ex)[:60]}...)", True)
        check("the refused bake left the target's weights as they were",
              (dst / "model.safetensors").read_bytes() == before)

        try:
            from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
        except ImportError:
            print("  skip derivation through a real speaker encoder: qwen_tts is not importable")
            return
        torch.manual_seed(0)
        enc_cfg = {"enc_dim": W, "enc_channels": [32, 32, 32, 32, 96], "enc_attention_channels": 16,
                   "enc_se_channels": 16, "sample_rate": SR}
        enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**enc_cfg))
        save_file({ENC_PREFIX + k: v.to(torch.bfloat16).contiguous() for k, v in enc.state_dict().items()},
                  str(enc_dir / "model.safetensors"), metadata={"format": "pt"})
        (enc_dir / "config.json").write_text(json.dumps({"tts_model_size": "0b6", "speaker_encoder_config": enc_cfg}),
                                             encoding="utf-8")
        rng = np.random.default_rng(0)
        clips = [td / "a.wav", td / "b.flac"]
        sf.write(str(clips[0]), (0.1 * rng.standard_normal(24000)).astype(np.float32), 24000)
        sf.write(str(clips[1]), (0.1 * rng.standard_normal(16000)).astype(np.float32), 16000)   # resampled to 24 kHz
        one = [encoder_mean(enc_dir, [c]) for c in clips]
        rec = bake(dst, src, {"hk_voice_343": 3000}, model_size="0b6", encoder=enc_dir,
                   audio={"hk_voice_343": [str(c) for c in clips]})
        row = load_file(str(dst / "model.safetensors"))[KEY]
        mean = torch.stack(one).mean(0).to(torch.bfloat16)
        check("the derived row is 1024 wide and is the encoder's mean over the clips",
              row.shape == (3072, W) and torch.equal(row[3000], mean) and rec["rows_equal_source"]["hk_voice_343"])
        check("RED: the derived row is not a single clip's vector", not torch.equal(row[3000], one[0].to(torch.bfloat16)))
        check("the record says the row was derived, and from which clips",
              rec["rows"]["hk_voice_343"]["from"] == "own-encoder mean" and len(rec["rows"]["hk_voice_343"]["clips"]) == 2)
        check("an unnamed row is untouched", torch.equal(row[2999], torch.full((W,), 2.0, dtype=torch.bfloat16)))
        trainer_dir = Path(__file__).resolve().parents[1] / "trainer"
        sys.path.insert(0, str(trainer_dir))
        try:
            from dataset import TTSDataset              # the trainer's own audio loading and mel, as the speaker phase runs it
        finally:
            sys.path.remove(str(trainer_dir))
        wav, sr = TTSDataset._load_audio_to_np(None, str(clips[0]))
        theirs = TTSDataset.extract_mels(None, audio=wav, sr=sr)
        ours = clip_mel(clips[0])
        check("on a 24 kHz clip the mel equals trainer/dataset.py extract_mels's",
              ours.shape == theirs.shape and torch.allclose(ours, theirs, rtol=0, atol=1e-6))
        try:
            TTSDataset.extract_mels(None, *TTSDataset._load_audio_to_np(None, str(clips[1])))
            check("the trainer refuses a 16 kHz clip (the documented difference: the bake resamples it)", False)
        except AssertionError:
            check("the trainer refuses a 16 kHz clip (the documented difference: the bake resamples it)", True)
        check("a 16 kHz clip is resampled to 24 kHz before the mel (1 s gives 24000 / 256 frames, not 16000 / 256)",
              abs(clip_mel(clips[1]).shape[1] - 24000 / 256) <= 2)
        same = td / "same"
        same.mkdir()
        save_file({KEY: torch.full((3072, W), 3.0, dtype=torch.bfloat16)}, str(same / "model.safetensors"),
                  metadata={"format": "pt"})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rec2 = bake(dst, same, {"hk_voice_343": 3000}, model_size="0b6", encoder=enc_dir,
                        audio={"hk_voice_343": [str(clips[0])]})
        check("clips given beside a same-width source are copied over, warned and recorded as ignored",
              "ignored" in err.getvalue() and rec2["rows"]["hk_voice_343"].get("clips_ignored") == [str(clips[0])]
              and torch.equal(load_file(str(dst / "model.safetensors"))[KEY][3000],
                              torch.full((W,), 3.0, dtype=torch.bfloat16)))
        try:
            bake(dst, None, {"hk_voice_343": 3000, "hk_voice_777": 3001}, model_size="0b6", encoder=enc_dir,
                 audio={"hk_voice_343": [str(clips[0])]})
            check("RED: a voice with no source and no clips is refused", False)
        except SystemExit:
            check("RED: a voice with no source and no clips is refused", True)
        wide = dict(enc_cfg, enc_dim=SRC_W)
        (enc_dir / "config.json").write_text(json.dumps({"tts_model_size": "1b7", "speaker_encoder_config": wide}),
                                             encoding="utf-8")
        wenc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**wide))
        save_file({ENC_PREFIX + k: v.to(torch.bfloat16).contiguous() for k, v in wenc.state_dict().items()},
                  str(enc_dir / "model.safetensors"), metadata={"format": "pt"})
        try:
            bake(dst, src, {"hk_voice_343": 3000}, model_size="0b6", encoder=enc_dir,
                 audio={"hk_voice_343": [str(clips[0])]})
            check("RED: an encoder of the other size (2048-wide vectors) is refused", False)
        except SystemExit:
            check("RED: an encoder of the other size (2048-wide vectors) is refused", True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=Path, help="the speaker phase's merged export (changed in place)")
    ap.add_argument("--source", type=Path, help="a merged checkpoint carrying the chosen voice rows")
    ap.add_argument("--voice", action="append", default=[], help="NAME=ROW, e.g. hk_voice_343=3000")
    ap.add_argument("--model-size", default="1b7", choices=MODEL_SIZES,
                    help="the target's tts_model_size; a config without one is refused (default 1b7)")
    ap.add_argument("--encoder", type=Path, help="a checkpoint of the target's size carrying speaker_encoder.*, to "
                                                 "derive rows a source of another width cannot give")
    ap.add_argument("--voice-audio", action="append", default=[],
                    help="NAME=CLIP, once per clip (a path may hold any character, commas included): the voice's "
                         "clips, for a row derived by the target's own encoder")
    ap.add_argument("--record", type=Path, help="write the record here too")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.target and (a.source or a.voice_audio) and a.voice):
        ap.error("--target, --source or --voice-audio, and at least one --voice are required")
    audio = {}
    for spec in a.voice_audio:
        name, clip = spec.split("=", 1)
        if not clip:
            ap.error(f"--voice-audio {spec}: no clip path")
        audio.setdefault(name, []).append(clip)
    if set(audio) - set(parse_voices(a.voice)):
        ap.error(f"--voice-audio names a voice that no --voice bakes: {sorted(set(audio) - set(parse_voices(a.voice)))}")
    rec = bake(a.target, a.source, parse_voices(a.voice), a.model_size, a.encoder, audio)
    print(json.dumps(rec, indent=1))
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        a.record.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
