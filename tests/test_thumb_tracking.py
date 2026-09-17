import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from thumb_tracking import ThumbTracker
from visual_pretouch import load_yaml


class ThumbTrackingTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_yaml(Path(__file__).resolve().parents[1] / 'config.yaml')['tracking']['continuity']
        self.tracker = ThumbTracker(self.settings)
        # A moving textured patch provides actual image motion, independent of model points.
        texture = np.random.default_rng(7).integers(40, 230, (45, 45, 3), dtype=np.uint8)
        self.frame = np.zeros((240, 400, 3), dtype=np.uint8)
        self.frame[75:120, 55:100] = texture
        for i in range(self.settings['reacquire_stable_frames']):
            self.tracker.update(self.frame, [78, 98], i * 17)
        self.time = self.settings['reacquire_stable_frames'] * 17

    def moved(self, dx):
        return cv2.warpAffine(self.frame, np.float32([[1, 0, dx], [0, 1, 0]]), (400, 240))

    def test_wrong_finger_does_not_replace_moving_thumb(self):
        point, source = self.tracker.update(self.moved(7), [280, 210], self.time)
        self.assertEqual(source, 'optical_flow')
        np.testing.assert_allclose(point, [85, 98], atol=1)

    def test_missing_model_can_use_current_image_motion(self):
        point, source = self.tracker.update(self.moved(7), None, self.time)
        self.assertEqual(source, 'optical_flow')
        np.testing.assert_allclose(point, [85, 98], atol=1)

    def test_flow_is_not_allowed_to_continue_indefinitely(self):
        point, source = self.tracker.update(self.frame, None, self.time + self.settings['max_flow_only_ms'])
        self.assertIsNone(point)
        self.assertEqual(source, 'missing')

    def test_image_loss_does_not_hold_previous_point(self):
        point, source = self.tracker.update(np.zeros_like(self.frame), None, self.time)
        self.assertIsNone(point)
        self.assertEqual(source, 'missing')

    def test_future_or_repeated_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tracker.update(self.frame, [78, 98], 0)


if __name__ == '__main__':
    unittest.main()
