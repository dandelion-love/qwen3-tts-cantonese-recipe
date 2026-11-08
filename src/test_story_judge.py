"""Plant a threshold drift against the recipe's actual story judge."""
import unittest
from unittest import mock
import numpy as np
import render_story as story


class StoryJudgeTests(unittest.TestCase):
    def test_threshold_and_planted_mutant(self):
        # 0.075 seconds per counted character lies strictly between 0.07 and 0.08.
        samples = np.ones(75, dtype="float32")
        import audio_trim
        def trim(y, sr, **kwargs):
            return y, {"start_sample": 0, "end_sample": len(y)}
        with mock.patch.object(audio_trim, "trim_edges", side_effect=trim):
            self.assertEqual(story.judge(samples, 1000, "好")[0], None)
            with mock.patch.object(story, "SILENT_SPC", 0.08):
                self.assertEqual(story.judge(samples, 1000, "好")[0], "silent")


if __name__ == "__main__":
    unittest.main()
