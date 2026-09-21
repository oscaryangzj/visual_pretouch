import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_session
import visual_pretouch as pipeline


class ReviewCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.state = {'session': 'session_test', 'width': 100, 'height': 100,
                      'orientation_degrees': 0, 'video_sha256': 'test-video',
                      'tracking': [[0., None], [.1, None], [1., None]],
                      'flashes': [.3], 'settings': {'after_flash_ms': 200},
                      'calibration': None,
                      'review': [{'trial_id': 'trial_0000', 'touch_index': '0', 'keep': '1'}]}
        self.request = {'frame_index': 1, 'image_points_px': [[10, 20], [90, 20], [90, 80], [10, 80]]}

    def test_mapping_uses_display_pixels_and_selected_frame(self):
        result = review_session.make_calibration(self.state, self.request)
        np.testing.assert_allclose(pipeline.project(np.asarray(result['homography_image_to_screen']),
                                                    self.request['image_points_px']),
                                   [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-6)
        self.assertEqual(result['reference_frame_index'], 1)
        self.assertEqual(result['reference_time_ms'], 100)
        with self.assertRaises(ValueError):
            review_session.make_calibration(self.state, {**self.request, 'frame_index': 2})

    def test_invalid_corner_order_or_location_is_rejected(self):
        for points in ([[[10, 20], [90, 80], [90, 20], [10, 80]],
                        [[10, 20], [10, 80], [90, 80], [90, 20]],
                        [[10, 20], [100, 20], [90, 80], [10, 80]],
                        [[10, 20], [90, 20], [90, float('nan')], [10, 80]]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                review_session.make_calibration(self.state, {**self.request, 'image_points_px': points})

    def test_api_saves_calibration_without_changing_existing_review(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            review = session / 'review.csv'
            review_session.save_review(review, self.state['review'])
            before = review.read_bytes()
            self.state.update(result_path=str(review), calibration_path=str(session / 'calibration.json'))
            server = ThreadingHTTPServer(('127.0.0.1', 0), review_session.make_handler(session / 'video.mp4', self.state))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            url = f'http://127.0.0.1:{server.server_port}/api/calibration'
            try:
                with urlopen(Request(url, json.dumps(self.request).encode(), {'Content-Type': 'application/json'})) as response:
                    saved = json.load(response)
                self.assertEqual(review.read_bytes(), before)
                path = session / 'calibration.json'
                self.assertEqual(review_session.load_calibration(path, self.state), saved['calibration'])
                self.assertEqual(saved['calibrations'][0]['source_index'], 0)
                unchanged = path.read_bytes()
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, json.dumps({**self.request, 'image_points_px': []}).encode()))
                self.assertEqual(error.exception.code, 400)
                self.assertEqual(path.read_bytes(), unchanged)
                self.assertEqual(review.read_bytes(), before)
                with self.assertRaises(ValueError):
                    review_session.load_calibration(path, {**self.state, 'video_sha256': 'other-video'})
            finally:
                server.shutdown()
                server.server_close()
                worker.join()

    def test_per_touch_inheritance_reannotation_and_reset_preserve_review(self):
        with tempfile.TemporaryDirectory() as directory:
            self.state.update(tracking=[[0., None], [.1, None], [.4, None], [.7, None], [1., None]],
                              flashes=[.3, .6, .9, 1.2],
                              review=[{'trial_id': f'trial_{i:04}', 'touch_index': str(i),
                                       'keep': '0' if i == 1 else '1'} for i in range(4)],
                              calibration_path=str(Path(directory) / 'calibration.json'))
            review = Path(directory) / 'review.csv'
            review_session.save_review(review, self.state['review'])
            before = review.read_bytes()
            review_session.update_calibration(self.state, self.request)
            self.assertEqual([r['source_index'] for r in self.state['calibrations']], [0, 0, 0, 0])
            changed = {**self.request, 'index': 2, 'frame_index': 3,
                       'image_points_px': [[20, 20], [90, 20], [90, 80], [20, 80]]}
            review_session.update_calibration(self.state, changed)
            self.assertEqual([r['source_index'] for r in self.state['calibrations']], [0, 0, 2, 2])
            self.assertTrue(self.state['calibrations'][3]['inherited'])
            # A new base must retain later annotations; nothing is propagated backwards.
            review_session.update_calibration(self.state, self.request)
            self.assertEqual(self.state['calibrations'][2]['calibration']['image_points_px'], changed['image_points_px'])
            review_session.update_calibration(self.state, {**changed, 'index': 3, 'frame_index': 4})
            path = Path(self.state['calibration_path'])
            loaded = review_session.load_calibration(path, self.state)
            self.assertEqual(review_session.effective_calibrations(loaded, self.state['review']), self.state['calibrations'])
            review_session.update_calibration(self.state, {'index': 2, 'inherit': True})
            self.assertEqual([r['source_index'] for r in self.state['calibrations']], [0, 0, 0, 3])
            self.assertEqual(review.read_bytes(), before)
            for request in ({**changed, 'index': 1, 'frame_index': 2},
                            {'index': 0, 'inherit': True}, {'index': 1, 'inherit': True},
                            {**changed, 'frame_index': 1}):
                unchanged = path.read_bytes()
                with self.assertRaises(ValueError):
                    review_session.update_calibration(self.state, request)
                self.assertEqual(path.read_bytes(), unchanged)

    def test_legacy_calibration_loads_without_rewriting_or_losing_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            base = review_session.make_calibration(self.state, self.request)
            base.pop('trial_id'); base.pop('touch_index')
            review_session.save_calibration(path, base)
            before = path.read_bytes()
            self.state['review'].append({'trial_id': 'trial_0001', 'touch_index': '1', 'keep': '0'})
            loaded = review_session.load_calibration(path, self.state)
            resolved = review_session.effective_calibrations(loaded, self.state['review'])
            self.assertEqual([r['source_index'] for r in resolved], [0, 0])
            self.assertEqual(resolved[1]['calibration']['image_points_px'], self.request['image_points_px'])
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
