import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import baseline_demo
import visual_pretouch as pipeline


class BaselineDemoTests(unittest.TestCase):
    def setUp(self):
        self.cfg = pipeline.load_yaml(Path(__file__).resolve().parents[1] / 'config.yaml')

    def test_velocity_ignores_future_points_and_does_not_bridge_long_gap(self):
        rows = [{'frame_index': i, 'video_time_ms': t, 'valid': 1,
                 'thumb_u': u, 'thumb_v': .4 + t * .001}
                for i, (t, u) in enumerate([(0, .3), (50, .4), (100, .5), (150, -5.)])]
        speed = baseline_demo.history_velocity(rows, rows[2], 0, self.cfg)
        np.testing.assert_allclose(speed, [.002, .001])
        self.cfg['tracking']['max_gap_ms'] = 40
        self.assertIsNone(baseline_demo.history_velocity(rows, rows[2], 0, self.cfg))
        self.assertIsNone(baseline_demo.history_velocity(rows, rows[2], 80, self.cfg))

    def test_b2_does_not_change_with_current_touch_time_b3_does(self):
        crossing = {'frame_index': 6, 'video_time_ms': 100., 'thumb_u': .51, 'thumb_v': .4}
        trial = {'session_id': 'test', 'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1',
                 'valid_gt': True, 'crossing': crossing, 'velocity': np.array([.003, -.002]),
                 'touch_time_video_ms': 200.}
        first = baseline_demo.predict_trial(trial, 50.)
        later = baseline_demo.predict_trial({**trial, 'touch_time_video_ms': 400.}, 50.)
        for i in (0, 1):
            self.assertEqual(first[i], later[i])
        self.assertNotEqual(first[2]['pred_u'], later[2]['pred_u'])
        self.assertEqual(later[2]['raw_pred_u'], .51 + .003 * 300)
        self.assertEqual(later[2]['pred_u'], 1)
        self.assertEqual(later[2]['pred_v'], 0)
        self.assertEqual(later[2]['clipped'], 1)
        self.assertEqual({r['prediction_frame'] for r in first}, {6})

    def test_velocity_scale_is_shared_by_b2_and_b3(self):
        crossing = {'frame_index': 6, 'video_time_ms': 100., 'thumb_u': .51, 'thumb_v': .4}
        trial = {'session_id': 'test', 'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1',
                 'valid_gt': True, 'crossing': crossing, 'velocity': np.array([.003, -.002]),
                 'touch_time_video_ms': 300.}
        rows = baseline_demo.predict_trial(trial, 100., .1)
        self.assertEqual(rows[0]['pred_u'], .51)
        np.testing.assert_allclose([rows[1]['raw_pred_u'], rows[1]['raw_pred_v']], [.54, .38])
        np.testing.assert_allclose([rows[2]['raw_pred_u'], rows[2]['raw_pred_v']], [.57, .36])
        for row in rows[1:]:
            self.assertEqual(row['velocity_scale'], .1)
            np.testing.assert_allclose([row['applied_velocity_u_per_ms'], row['applied_velocity_v_per_ms']],
                                       [.0003, -.0002])
        self.assertEqual(rows[0]['velocity_scale'], '')

    def test_anchor_is_observed_point_and_failed_predictions_stay_blank(self):
        raw, clipped = baseline_demo.predict_position([.51, .4], np.array([.002, 0.]), 100.)
        np.testing.assert_allclose(raw, [.71, .4])
        np.testing.assert_allclose(clipped, raw)
        trial = {'session_id': 'test', 'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1',
                 'valid_gt': True, 'crossing': {'frame_index': 6, 'video_time_ms': 100., 'thumb_u': .51, 'thumb_v': .4},
                 'velocity': None, 'touch_time_video_ms': 200.}
        rows = baseline_demo.predict_trial(trial, None)
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertTrue(all(r['pred_u'] == r['pred_v'] == r['clipped'] == '' for r in rows[1:]))
        _, metrics = baseline_demo.evaluate(rows, [{**trial, 'gt_u': .8, 'gt_v': .4}],
                                            {'region_width_mm': 70, 'region_height_mm': 150}, self.cfg['evaluation'])
        self.assertEqual(metrics['methods'][baseline_demo.METHODS[1]]['hit_at_20mm_all_kept'], 0)

    def test_each_action_uses_its_own_mapping_for_crossing_and_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            for name in ('review.csv', 'calibration.json', '.review_cache.json'):
                (session / name).write_text('fixture')
            touches = [{'trial_id': f'trial_{i:04}', 'touch_index': str(i), 'touch_time_browser_ms': 900 + 1000 * i,
                        'touch_u': .8, 'touch_v': .5} for i in range(2)]
            pipeline.write_csv(session / 'fixture_touches.csv', touches, list(touches[0]))
            reviews = [{**t, 'keep': '1'} for t in touches]
            calibrations = []
            for offset in (10, 110):
                points = np.float32([[offset, 100], [offset + 100, 100], [offset + 100, 280], [offset, 280]])
                H = cv2.getPerspectiveTransform(points, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
                calibrations.append({'image_points_px': points.tolist(), 'homography_image_to_screen': H.tolist()})
            tracking = [[i * .05, [10 + i * 5, 190] if i <= 18 else
                         [110 + (i - 20) * 5, 190] if i >= 24 else None] for i in range(45)]
            state = {'tracking': tracking, 'review': reviews, 'flashes': [.9, 1.9],
                     'calibrations': [{'calibration': cal, 'source_index': i, 'inherited': False}
                                      for i, cal in enumerate(calibrations)]}
            with patch.object(baseline_demo.review_session, 'prepare', return_value=(session / 'video.mp4', state)):
                result = baseline_demo.prepare_session(session, self.cfg)
            for i, trial in enumerate(result['trials']):
                self.assertAlmostEqual(trial['crossing']['video_time_ms'], 550 + i * 1000)
                self.assertEqual(trial['calibration_source_trial_id'], touches[i]['trial_id'])
                np.testing.assert_allclose(trial['velocity'], [.001, 0], atol=1e-9)
                action = [r for r in result['projected_trajectory'] if r['trial_id'] == trial['trial_id']]
                self.assertTrue(all(r['calibration_source_trial_id'] == trial['trial_id'] for r in action))

    def test_video_prediction_starts_at_crossing_and_gt_only_after_click(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.mp4', Path(directory) / 'demo.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (200, 300))
            self.assertTrue(writer.isOpened())
            for _ in range(11):
                writer.write(np.zeros((300, 200, 3), np.uint8))
            writer.release()
            points = np.float32([[30, 150], [170, 150], [170, 290], [30, 290]])
            H = cv2.getPerspectiveTransform(points, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
            trajectory = [{'frame_index': i, 'video_time_ms': i * 100., 'valid': 0} for i in range(11)]
            trial = {'session_id': 'test', 'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1',
                     'valid_gt': True, 'crossing': {'frame_index': 5, 'video_time_ms': 500., 'thumb_u': .51, 'thumb_v': .4},
                     'velocity': None, 'touch_time_video_ms': 800., 'action_start_ms': 0., 'gt_u': .8, 'gt_v': .7,
                     'calibration': {'homography_image_to_screen': H.tolist(), 'image_points_px': points.tolist()}}
            details, _ = baseline_demo.evaluate(baseline_demo.predict_trial(trial, None), [trial], {}, self.cfg['evaluation'])
            data = {'video': source, 'trajectory': trajectory,
                    'state': {'width': 200, 'height': 300, 'fps': 10, 'duration': 1.1,
                              'calibration': {'homography_image_to_screen': np.eye(3).tolist(),
                                              'image_points_px': [[0, 0], [1, 0], [1, 1], [0, 1]]}}}
            result = baseline_demo.render(data, [trial], details, baseline_demo.METHODS[0], output, self.cfg['render'])
            self.assertEqual(result['frames'], 11)
            cap = cv2.VideoCapture(str(output))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame.astype(np.int16))
            cap.release()
            def red(frame):
                region = frame[190:220, 90:112]
                return np.any((region[:, :, 2] - region[:, :, 1] > 100) & (region[:, :, 2] - region[:, :, 0] > 100))
            def blue(frame):
                region = frame[230:265, 125:159]
                return np.any((region[:, :, 0] - region[:, :, 1] > 80) & (region[:, :, 0] - region[:, :, 2] > 100))
            self.assertFalse(red(frames[4]))
            self.assertTrue(red(frames[5]))
            self.assertFalse(blue(frames[7]))
            self.assertTrue(blue(frames[8]))


if __name__ == '__main__':
    unittest.main()
