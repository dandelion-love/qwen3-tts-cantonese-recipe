"""CPU-only checks for the runtime stamp and story manifest."""
from __future__ import annotations
import hashlib
import subprocess
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_story as rs
import gguf_backend as gb


class Provenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.root = root
        script = root / 'scripts' / 'build_gguf_runtime.sh'
        script.parent.mkdir()
        script.write_text('QWEN_TTS_CPP_PIN:-' + 'a' * 40, encoding='utf8')
        patches = root / 'external' / 'qwen3-tts-cpp' / 'patches'
        patches.mkdir(parents=True)
        (patches / 'synthetic.patch').write_bytes(b'patch')
        self.cli = root / 'qwen3-tts-cli'
        self.cli.write_bytes(b'binary')
        self.codec = root / 'codec.gguf'
        self.codec.write_bytes(b'codec')
        self.stamp = {'version': 1, 'pin_built': 'a' * 40, 'ggml_commit': 'b' * 40,
                      'patches_built': {'synthetic.patch': hashlib.sha256(b'patch').hexdigest()},
                      'cli_sha256': hashlib.sha256(b'binary').hexdigest()}
        self.write_stamp()
        self.stack = mock.patch.multiple(rs, RECIPE_ROOT=root)
        self.stack.start()
        self.addCleanup(self.stack.stop)
        self.cli_patch = mock.patch.object(gb, 'CLI_BIN', self.cli)
        self.cli_patch.start()
        self.addCleanup(self.cli_patch.stop)
        self.codec_patch = mock.patch.object(gb, '_find_codec', return_value=self.codec)
        self.codec_patch.start()
        self.addCleanup(self.codec_patch.stop)

    def write_stamp(self):
        Path(str(self.cli) + '.build.json').write_text(json.dumps(self.stamp), encoding='utf8')

    def manifest(self):
        return {'rule': {'seed_rule': rs.SEED_RULE}, 'engine': rs.engine_block()}

    def test_valid_renderer_block_and_missing_fields(self):
        man = self.manifest()
        self.assertEqual(rs.provenance_problems(man), [])
        for field in ('backend', 'pin_expected', 'pin_built', 'ggml_commit', 'cli_sha256',
                      'patches_expected', 'patches_built', 'codec', 'codec_sha256'):
            candidate = json.loads(json.dumps(man))
            del candidate['engine'][field]
            self.assertTrue(any(field in x for x in rs.provenance_problems(candidate)), field)
        with self.assertRaises(SystemExit):
            rs.require_provenance({'engine': man['engine'], 'rule': {}}, 'test')
        self.assertTrue(rs.provenance_problems({'rule': man['rule']}))

    def test_absent_malformed_stale_and_replaced_binary_refused(self):
        stamp_path = Path(str(self.cli) + '.build.json')
        for content in (None, 'invalid', '{}'):
            if content is None:
                stamp_path.unlink(missing_ok=True)
            else:
                stamp_path.write_text(content, encoding='utf8')
            self.assertTrue(rs.provenance_problems(self.manifest()))
        self.write_stamp()
        self.cli.write_bytes(b'replaced')
        self.assertTrue(rs.provenance_problems(self.manifest()))

    def test_build_script_publishes_only_matching_patch_snapshot(self):
        script = (Path(__file__).resolve().parents[1] / 'scripts/build_gguf_runtime.sh').read_text(encoding='utf8')
        marker = 'python3 - "$CLI" "$SRC" "$PATCHES" "$PATCH_SNAPSHOT" "$SOURCE_HEAD" <<\'PYCODE\'\n'
        source = script.split(marker, 1)[1].split('\nPYCODE', 1)[0]
        patches = self.root / 'external/qwen3-tts-cpp/patches'
        snapshot = self.root / 'snapshot.json'
        snapshot.write_text(json.dumps(self.stamp['patches_built']), encoding='utf8')
        def git_output(args, text):
            return ('160000 commit ' + 'b' * 40 + '\tggml\n') if 'ls-tree' in args else 'a' * 40 + '\n'
        args = ['stamp', str(self.cli), str(self.root), str(patches), str(snapshot), 'a' * 40]
        stamp_path = Path(str(self.cli) + '.build.json')
        stamp_path.unlink()
        with mock.patch.object(sys, 'argv', args), mock.patch.object(subprocess, 'check_output', side_effect=git_output):
            exec(compile(source, 'build_gguf_runtime.sh:stamp', 'exec'), {})
            self.assertEqual(json.loads(stamp_path.read_text()), self.stamp)
            stamp_path.unlink()
            (patches / 'synthetic.patch').write_bytes(b'changed')
            with self.assertRaises(SystemExit):
                exec(compile(source, 'build_gguf_runtime.sh:stamp', 'exec'), {})
            self.assertFalse(stamp_path.exists())

    def test_mock_build_producer_and_failures(self):
        # WSL tmp only: no model, network, actual compiler or active runtime.
        harness = r'''
set -eu
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
mkdir -p "$root/repo/scripts" "$root/repo/external/qwen3-tts-cpp/patches" "$root/src/.git" "$root/bin" "$root/build"
cp scripts/build_gguf_runtime.sh "$root/repo/scripts/"
touch "$root/repo/recipe.toml"
printf patch > "$root/repo/external/qwen3-tts-cpp/patches/a.patch"
printf aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa > "$root/head"
export MOCK_ROOT="$root" QWEN_TTS_CPP="$root/src" QWEN_TTS_CPP_BUILD="$root/build"
export QWEN_TTS_CUDA_OFF=1 QWEN_TTS_CPP_PIN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
export PATH="$root/bin:$PATH"
cat > "$root/bin/git" <<'GIT'
#!/bin/bash
if [ "$1" = -C ]; then shift 2; fi
case "$1 $2" in
  'rev-parse HEAD') cat "$MOCK_ROOT/head" ;;
  'rev-parse '*) printf '%s\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
  'diff --binary') printf source-diff ;;
  'ls-tree '*) printf '160000 commit bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\tggml\n' ;;
  *) exit 0 ;;
esac
GIT
cat > "$root/bin/cmake" <<'CMAKE'
#!/bin/bash
if [ "$1" = --build ]; then
  if [ "${MOCK_MODE:-}" = fail ]; then exit 1; fi
  cat > "$QWEN_TTS_CPP_BUILD/qwen3-tts-cli" <<'CLI'
#!/bin/bash
echo '--serve ;codes=<path>'
CLI
  chmod +x "$QWEN_TTS_CPP_BUILD/qwen3-tts-cli"
  case "${MOCK_MODE:-}" in
    source) printf cccccccccccccccccccccccccccccccccccccccc > "$MOCK_ROOT/head" ;;
    patch) printf changed > "$MOCK_ROOT/repo/external/qwen3-tts-cpp/patches/a.patch" ;;
  esac
fi
CMAKE
chmod +x "$root/bin/git" "$root/bin/cmake"
run_build() { bash "$root/repo/scripts/build_gguf_runtime.sh" >"$root/build.log" 2>&1; }
stamp="$root/build/qwen3-tts-cli.build.json"
run_build || { cat "$root/build.log"; exit 1; }
python3 - "$stamp" "$root/build/qwen3-tts-cli" <<'CHECK'
import hashlib, json, pathlib, sys
stamp = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert stamp['cli_sha256'] == hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
assert stamp['pin_built'] == 'a' * 40
assert stamp['ggml_commit'] == 'b' * 40
assert stamp['version'] == 1
assert stamp['patches_built']['a.patch'] == hashlib.sha256(b'patch').hexdigest()
CHECK
for mode in fail source patch; do
  printf aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa > "$root/head"
  printf patch > "$root/repo/external/qwen3-tts-cpp/patches/a.patch"
  printf stale > "$stamp"
  if MOCK_MODE="$mode" run_build; then echo "unexpected success: $mode" >&2; exit 1; fi
  test ! -e "$stamp" || { echo "stale stamp remains: $mode" >&2; exit 1; }
done
'''
        result = subprocess.run(["bash", "-s"], input=harness.encode(), capture_output=True)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors="replace"))

    def test_checkpoint_card_refuses_missing_backend(self):
        # Legacy manifests without a stamped runtime identity are also refused;
        # re-render with a verified build rather than infer historic identity.
        import checkpoint_card as card
        candidate = self.manifest()
        del candidate['engine']['backend']
        (self.root / 'manifest.json').write_text(json.dumps(candidate), encoding='utf8')
        modules = {name: types.ModuleType(name) for name in ('skipped_text', 'steps', 'audio_uri')}
        modules['audio_uri'].write_flac = lambda *args: None
        with mock.patch.dict(sys.modules, modules), mock.patch.object(card, '_ASR', {'q': object(), 's': object()}):
            with self.assertRaisesRegex(SystemExit, 'backend'):
                card.dropped(self.root, 'fixture')

    def test_wrong_pin_patch_or_ggml_refused(self):
        for field, value in (('pin_built', 'c' * 40), ('patches_built', {}), ('ggml_commit', None)):
            old = self.stamp[field]
            self.stamp[field] = value
            self.write_stamp()
            self.assertTrue(rs.provenance_problems(self.manifest()), field)
            self.stamp[field] = old
        self.write_stamp()
        (self.root / 'external/qwen3-tts-cpp/patches/synthetic.patch').write_bytes(b'changed')
        self.assertTrue(rs.provenance_problems(self.manifest()))


if __name__ == '__main__':
    unittest.main()
