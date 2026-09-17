import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_prepare
import review_session


class ReviewPreparationTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'flash_rise_thresholds_gray': [2, 4],
                         'flash_min_interval_ms': 200, 'flash_max_residual_ms': 30}
        self.times = list(range(0, 2200, 100))
        self.light = [10.] * len(self.times)
        for i in (2, 12):
            self.light[i:i+2] = [15., 18.]
        self.touches = [{'touch_time_browser_ms': '5000'}, {'touch_time_browser_ms': '6000'}]

    def test_pairing_uses_relative_timing_and_merges_flash_rise(self):
        events, details = review_prepare.match_flashes(self.times, self.light, self.touches, self.settings)
        self.assertEqual([e['frame_index'] for e in events], [2, 12])
        self.assertEqual(details['max_residual_ms'], 0)

    def test_wrong_timing_rejected_even_with_matching_count(self):
        self.touches[1]['touch_time_browser_ms'] = '6400'
        with self.assertRaises(ValueError):
            review_prepare.match_flashes(self.times, self.light, self.touches, self.settings)

    def test_extra_flash_not_silently_truncated(self):
        self.light[18] = 20
        with self.assertRaises(ValueError):
            review_prepare.match_flashes(self.times, self.light, self.touches, self.settings)

    def test_generic_file_names_and_single_result_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            touches = [{'trial_id': 'trial_0000', 'touch_index': '0'}]
            (session / 'touches.csv').write_text('trial_id,touch_index\ntrial_0000,0\n')
            self.assertEqual(review_session.one_file(session, '*_touches.csv').name, 'touches.csv')
            rows = review_session.load_review(session, touches)
            rows[0]['keep'] = '0'
            review_session.save_review(session / 'review.csv', rows)
            self.assertEqual(review_session.load_review(session, touches)[0]['keep'], '0')
            self.assertEqual(sorted(p.name for p in session.iterdir()), ['review.csv', 'touches.csv'])

    def test_unversioned_tracking_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            old = session / 'frame_diagnostics.csv'
            old.write_text('old lightweight results')
            self.assertIsNone(review_session.source_path(session, None, old.name))
            self.assertEqual(review_session.source_path(session, old, old.name), old.resolve())


if __name__ == '__main__':
    unittest.main()
