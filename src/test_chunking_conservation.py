"""Regression checks for story text and punctuation conservation."""
import sys
import unicodedata
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk


class ChunkConservation(unittest.TestCase):
    def test_previously_lost_fragments_are_retained(self):
        cases = ('a', 'ab\u3002c', '\uff01\uff1fhello\u3002',
                 'hello\uff1f\uff01world\u3002', '\u300chello\u3002\u300d',
                 'abcdefghijklmnopqrstuvwxyza\u3002b', 'abcdefghijklmnopqrstuvwxyz1')
        for text in cases:
            with self.subTest(text=text):
                pieces = chunk(text, 14, 26)
                self.assertEqual(''.join(body + mark for body, mark in pieces), text)
                for body, mark in pieces:
                    self.assertTrue(any(not unicodedata.category(c).startswith('P')
                                        and not c.isspace() for c in body + mark))

    def test_unchanged_results_when_no_text_was_lost(self):
        for text, expected in (
            ('abcd', [('abcd', '')]),
            ('abcd\u3002', [('abcd', '\u3002')]),
            ('abcd\uff0cefgh\u3002', [('abcd\uff0cefgh', '\u3002')]),
            ('abcd\u3002efgh\uff1f', [('abcd\u3002efgh', '\uff1f')]),
        ):
            with self.subTest(text=text):
                self.assertEqual(chunk(text, 14, 26), expected)

    def test_punctuation_only_cannot_be_model_request(self):
        for text in ('\u3002', '\u300c\uff01\u300d', '\uff1f' * 30 + 'a'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                chunk(text, 14, 26)
        self.assertEqual(chunk('  ', 14, 26), [])

    def test_punctuation_pause_and_real_story_join(self):
        import numpy as np
        import audio_trim
        import gguf_backend
        import render_story as story
        for marks, expected in (("\uff1f\uff01", 0.50), ("\u3002\u300d", 0.48),
                                ("\uff0c\uff01", 0.20), ("\uff1f", 0.50), ("", 0.18)):
            with self.subTest(marks=marks):
                self.assertEqual(story.punctuation_pause(marks, story.PAUSE, story.PAUSE[""]), expected)

        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.requests = []
            def generate_custom_voice(self, **kwargs):
                self.requests.append(kwargs['text'][0])
                return [np.ones(10, dtype='float32')], 1000
            def close(self):
                pass

        plan = story.plan(['abcdefghij\uff1f\uff01klmnopqrst\u3002'], hi=12)
        with mock.patch.object(gguf_backend, 'GGUFModel', FakeModel), \
             mock.patch.object(story, 'judge', return_value=(None, 0.1)), \
             mock.patch.object(audio_trim, 'trim_edges', side_effect=lambda y, sr, **kw: (y, {})):
            result = story.render_arm('fixture', Path('synthetic'), 'voice', {'s': ('title', plan)}, Path('.'))
        rows = result['s']['chunks']
        self.assertEqual(rows[1]['span'][0] - rows[0]['span'][1], 500)
        self.assertEqual([row['text'] for row in rows], [item['text'] for item in plan])

    def test_story_plan_passes_each_surviving_character_once(self):
        import render_story
        paragraphs = ['\uff01hello\uff1f\uff01world\u3002', '\u300cabcd\u3002\u300d']
        planned = render_story.plan(paragraphs, hi=26)
        self.assertEqual(''.join(c['text'] for c in planned), ''.join(paragraphs))
        self.assertEqual([c['para'] for c in planned], [0, 1])
        with self.assertRaises(ValueError):
            render_story.plan(['\uff1f'], hi=26)


if __name__ == '__main__':
    unittest.main()
