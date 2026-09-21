import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import visual_pretouch as pipeline
import m2_endpoint_models as m2


class M2EndpointTests(unittest.TestCase):
    def setUp(self):
        self.feature_cfg = {
            'history_windows_ms': [50, 100, 200, 300],
            'quality_window_ms': 300,
            'min_points_per_window': 3,
            'group_cv_folds': 2,
            'random_state': 42,
            'ridge_alphas': [1.0],
            'gradient_boosting': {
                'loss': 'huber', 'n_estimators': [10], 'learning_rate': [0.05],
                'max_depth': [1], 'min_samples_leaf': 1,
            },
        }

    def test_features_stop_at_crossing(self):
        trial = {
            'session_id': 'train_a', 'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1',
            'action_start_ms': 0.0, 'gt_u': 0.8, 'gt_v': 0.3, 'valid_gt': True,
            'crossing': {'frame_index': 2, 'video_time_ms': 100.0, 'thumb_u': 0.55, 'thumb_v': 0.2},
        }
        projected = []
        for index, (time, u, v) in enumerate(((0, 0.40, 0.10), (50, 0.48, 0.15),
                                                (100, 0.55, 0.20), (150, 0.99, 0.99))):
            projected.append({'trial_id': 'trial_0000', 'touch_index': '0', 'video_time_ms': time,
                              'thumb_u': u, 'thumb_v': v, 'valid': 1, 'frame_index': index})
        data = {'projected_trajectory': projected, 'tracking_max_gap_ms': 100.0}
        sample = m2.sample_from_trial(data, trial, self.feature_cfg, 'train', 'device',
                                      {'region_width_mm': 70.0, 'region_height_mm': 150.0})
        self.assertEqual(sample['max_feature_time_ms'], sample['prediction_time_ms'])
        self.assertEqual(sample['crossing_u'], 0.55)
        self.assertAlmostEqual(sample['displacement_u_100ms'], 0.15)
        self.assertLess(sample['path_length'], 0.3)
        self.assertFalse(math.isclose(sample['path_length'], math.sqrt(2), rel_tol=1e-6))

    def test_grouped_cv_never_shares_session_between_folds(self):
        names = ['crossing_u', 'crossing_v', 'action_elapsed_ms']
        samples = []
        for session_index, session_id in enumerate(('a', 'b', 'c', 'd')):
            for row_index in range(2):
                samples.append({
                    'session_id': session_id,
                    'crossing_u': 0.51 + 0.01 * row_index,
                    'crossing_v': 0.3 + 0.01 * session_index,
                    'action_elapsed_ms': 100 + row_index,
                    'delta_u': 0.1 + 0.01 * session_index,
                    'delta_v': 0.05 + 0.01 * row_index,
                    'region_width_mm': 70.0,
                    'region_height_mm': 150.0,
                })
        _, selection = m2.select_model('ridge', samples, names, self.feature_cfg)
        for candidate in selection['candidate_results']:
            for fold in candidate['folds']:
                self.assertTrue(set(fold['train_sessions']).isdisjoint(fold['validation_sessions']))

    def test_median_prediction_is_anchored_and_clipped(self):
        sample = {
            'session_id': 'test', 'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1',
            'split': 'test', 'prediction_frame': 10, 'prediction_time_ms': 100.0,
            'max_feature_time_ms': 100.0, 'eligible': 1, 'exclusion_reason': '',
            'crossing_u': 0.9, 'crossing_v': 0.8,
        }
        class DummyModel:
            def predict(self, _features):
                return np.asarray([[0.3, 0.4]])

        rows = m2.predict_sample(sample, np.asarray([0.3, 0.4]),
                                 {m2.METHODS[1]: DummyModel(), m2.METHODS[2]: DummyModel()}, [])
        row = rows[0]
        self.assertEqual(row['status'], 'ok')
        self.assertEqual(row['clipped'], 1)
        self.assertAlmostEqual(row['raw_pred_u'], 1.2)
        self.assertAlmostEqual(row['raw_pred_v'], 1.2)
        self.assertEqual((row['pred_u'], row['pred_v']), (1.0, 1.0))

    def test_test_split_is_not_training_split(self):
        cfg = pipeline.load_yaml(Path(__file__).resolve().parents[1] / 'config.yaml')
        train = set(cfg['data_split']['training_sessions'])
        test = set(cfg['data_split']['test_sessions'])
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertTrue(train.isdisjoint(test))


if __name__ == '__main__':
    unittest.main()
