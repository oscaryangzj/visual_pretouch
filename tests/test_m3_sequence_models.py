import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m3_sequence_models as m3


class M3SequenceTests(unittest.TestCase):
    def test_interpolation_does_not_bridge_long_gap(self):
        points = [
            {'video_time_ms': 0.0, 'thumb_u': 0.4, 'thumb_v': 0.2},
            {'video_time_ms': 120.0, 'thumb_u': 0.6, 'thumb_v': 0.3},
        ]
        self.assertIsNone(m3.interpolate_point(points, 60.0, 100.0))
        np.testing.assert_allclose(m3.interpolate_point(points, 60.0, 120.0), [0.5, 0.25])

    def test_sequence_is_causal_and_ends_at_crossing(self):
        trial = {
            'trial_id': 'trial_0000', 'touch_index': '0', 'action_start_ms': 0.0,
            'crossing': {'video_time_ms': 100.0, 'thumb_u': 0.55, 'thumb_v': 0.4},
        }
        projected = [
            {'trial_id': 'trial_0000', 'touch_index': '0', 'video_time_ms': 0.0,
             'thumb_u': 0.4, 'thumb_v': 0.3, 'valid': 1},
            {'trial_id': 'trial_0000', 'touch_index': '0', 'video_time_ms': 50.0,
             'thumb_u': 0.48, 'thumb_v': 0.35, 'valid': 1},
            {'trial_id': 'trial_0000', 'touch_index': '0', 'video_time_ms': 100.0,
             'thumb_u': 0.55, 'thumb_v': 0.4, 'valid': 1},
            {'trial_id': 'trial_0000', 'touch_index': '0', 'video_time_ms': 150.0,
             'thumb_u': 0.99, 'thumb_v': 0.99, 'valid': 1},
        ]
        sample = {'crossing_u': 0.55, 'crossing_v': 0.4, 'action_elapsed_ms': 100.0}
        sequence, mask, static = m3.build_sequence(
            {'projected_trajectory': projected, 'tracking_max_gap_ms': 100.0}, trial, sample,
            {'sequence_window_ms': 100.0, 'sequence_points': 3})
        np.testing.assert_allclose(sequence[-1, :2], [0.0, 0.0])
        self.assertEqual(mask[-1], 1.0)
        np.testing.assert_allclose(static, [0.55, 0.4, 100.0])
        self.assertLess(sequence[1, 0], 0.0)

    def test_tiny_models_have_expected_size_and_output(self):
        import torch
        lstm = m3.TinySequenceRegressor('lstm', 3, 16, 3)
        gru = m3.TinySequenceRegressor('gru', 3, 16, 3)
        self.assertEqual(sum(parameter.numel() for parameter in lstm.parameters()), 1384)
        self.assertEqual(sum(parameter.numel() for parameter in gru.parameters()), 1048)
        sequence = torch.zeros((2, 10, 3))
        static = torch.zeros((2, 3))
        self.assertEqual(tuple(lstm(sequence, static).shape), (2, 2))
        self.assertEqual(tuple(gru(sequence, static).shape), (2, 2))


if __name__ == '__main__':
    unittest.main()
