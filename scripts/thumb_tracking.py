"""Causal thumb identity tracking; never interpolate from future frames."""

import cv2
import numpy as np


class ThumbTracker:
    def __init__(self, settings):
        self.settings = settings
        self.previous = None
        self.point = None
        self.anchor = None
        self.candidate = None
        self.stable = 0
        self.confirmed_ms = None
        self.last_time_ms = None

    def update(self, frame, measurement, time_ms):
        if self.last_time_ms is not None and time_ms <= self.last_time_ms:
            raise ValueError('Thumb tracking requires increasing timestamps')
        self.last_time_ms = time_ms
        s = self.settings
        if not s.get('enabled', False):
            return measurement, 'mediapipe' if measurement is not None else 'missing'
        height = min(frame.shape[0], int(s['input_height_px']))
        width = round(frame.shape[1] * height / frame.shape[0])
        gray = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY)
        scale = np.array([frame.shape[1] / width, frame.shape[0] / height], dtype=np.float32)
        raw = None if measurement is None else np.array(measurement, dtype=np.float32) / scale
        if raw is not None and (not np.isfinite(raw).all() or
                                not (0 <= raw[0] < width and 0 <= raw[1] < height)):
            raw = None
        estimate = None
        if self.previous is not None and self.point is not None:
            params = dict(winSize=(s['lk_window_px'], s['lk_window_px']),
                          maxLevel=s['lk_max_level'],
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                    s['lk_iterations'], s['lk_epsilon']))
            start = self.point.astype(np.float32).reshape(1, 1, 2)
            moved, ok, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, start, None, **params)
            if moved is not None and ok[0, 0] and error[0, 0] <= s['max_lk_error']:
                back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, **params)
                x, y = moved.reshape(2)
                if (back is not None and back_ok[0, 0] and
                    np.linalg.norm(back - start) <= s['max_forward_backward_error_px'] and
                    0 <= x < width and 0 <= y < height):
                    estimate = moved.reshape(2)
        source = 'missing'
        self.point = None
        if estimate is not None:
            if raw is not None and np.linalg.norm(raw - estimate) <= s['model_match_radius_px']:
                self.point = raw
                self.confirmed_ms = time_ms
                source = 'mediapipe'
            elif time_ms - self.confirmed_ms <= s['max_flow_only_ms']:
                self.point = estimate
                source = 'optical_flow'
        if self.point is None:
            # After losing the image track, a distant fingertip must not take its identity.
            near = raw is not None and (self.anchor is None or
                   np.linalg.norm(raw - self.anchor) <= s['reacquire_radius_px'])
            if near:
                consistent = self.candidate is not None and np.linalg.norm(raw - self.candidate) <= s['model_match_radius_px']
                self.stable = self.stable + 1 if consistent else 1
                self.candidate = raw
                if self.stable >= s['reacquire_stable_frames']:
                    self.point = raw
                    self.confirmed_ms = time_ms
                    source = 'mediapipe'
            else:
                self.stable = 0
                self.candidate = None
        else:
            self.stable = 0
            self.candidate = None
        if self.point is not None:
            self.anchor = self.point.copy()
        self.previous = gray
        return (None if self.point is None else (self.point * scale).tolist()), source
