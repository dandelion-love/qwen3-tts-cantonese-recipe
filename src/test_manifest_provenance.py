# SPDX-License-Identifier: Apache-2.0
"""The story manifest says what made the read, and a reader refuses one that does not.

Runs on CPU with the standard library only; nothing is rendered:

    python src/test_manifest_provenance.py

Covers render_story.provenance_problems and require_provenance on a complete manifest and on one missing each field,
and render_story.engine_block() on this checkout with a stand-in codec file, so the block the renderer writes is the
block the reader accepts.

NOT covered: a real render. The engine block's pin comes from scripts/build_gguf_runtime.sh here; on a machine with a
built engine, pin_built and cli_sha256 are filled from it too.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_story as rs  # noqa: E402

FULL = {"rule": {"seed_rule": rs.SEED_RULE},
        "engine": {"backend": "gguf", "pin_built": "a" * 40, "pin_expected": "a" * 40,
                   "patches": {"0001-serve-resident-mode.patch": "b" * 64}, "codec_sha256": "c" * 64}}


def without(*path) -> dict:
    m = json.loads(json.dumps(FULL))
    d = m
    for k in path[:-1]:
        d = d[k]
    del d[path[-1]]
    return m


class Provenance(unittest.TestCase):
    def test_a_complete_manifest_passes(self):
        self.assertEqual(rs.provenance_problems(FULL), [])

    def test_each_missing_field_is_refused(self):
        no_pins = json.loads(json.dumps(FULL))
        del no_pins["engine"]["pin_built"], no_pins["engine"]["pin_expected"]
        for name, m in (("the backend", without("engine", "backend")), ("the engine block", without("engine")),
                        ("the codec hash", without("engine", "codec_sha256")), ("both pins", no_pins),
                        ("the patches", without("engine", "patches")), ("the seed rule", without("rule", "seed_rule"))):
            self.assertTrue(rs.provenance_problems(m), f"a manifest without {name} was accepted")

    def test_require_provenance_stops_and_names_the_field(self):
        with self.assertRaises(SystemExit) as e:
            rs.require_provenance(without("engine", "backend"), "planted")
        self.assertIn("backend", str(e.exception))

    def test_the_block_the_renderer_writes_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            codec = Path(d) / "codec.gguf"
            codec.write_bytes(b"stand-in codec")
            import gguf_backend as gb
            with mock.patch.object(gb, "_find_codec", return_value=codec):
                eng = rs.engine_block()
        self.assertEqual(eng["backend"], "gguf")
        self.assertRegex(eng["pin_expected"] or "", r"^[0-9a-f]{40}$", "the build script's pin was not read")
        self.assertIn("0001-serve-resident-mode.patch", eng["patches"])
        self.assertEqual(rs.provenance_problems({"rule": {"seed_rule": rs.SEED_RULE}, "engine": eng}), [])


if __name__ == "__main__":
    unittest.main()
