"""Regression checks for story text and punctuation conservation."""
import sys
import unicodedata
import unittest
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
