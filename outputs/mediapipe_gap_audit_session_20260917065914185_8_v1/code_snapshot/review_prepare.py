"""Prepare reusable image-coordinate tracking for any collection session."""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

import visual_pretouch as pipeline
from thumb_tracking import ThumbTracker


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def match_flashes(times, brightness, touches, settings):
    """Require both count and relative click timing; never truncate events."""
    browser = np.array([float(t['touch_time_browser_ms']) for t in touches])
    if not np.isfinite(browser).all() or np.any(np.diff(browser) <= 0):
        raise ValueError('Touch timestamps must be finite and increasing')
    rises = np.diff(brightness, prepend=brightness[0])
    candidates = []
    for threshold in settings['flash_rise_thresholds_gray']:
        indices = []
        for i in np.flatnonzero(rises >= threshold):
            if not indices or times[i] - times[indices[-1]] >= settings['flash_min_interval_ms']:
                indices.append(int(i))
        if len(indices) != len(touches):
            continue
        offsets = np.array([times[i] for i in indices]) - browser
        residual = float(np.max(np.abs(offsets - np.median(offsets))))
        if residual <= settings['flash_max_residual_ms']:
            candidates.append((residual, indices, threshold))
    if not candidates:
        raise ValueError('Cannot reliably pair flashes with touches. Tracking is cached. '
                         'Set review.preparation.flash_roi_uv in config.yaml to a visible screen patch, '
                         'then rerun, or provide --events PATH to a checked alignment.csv.')
    residual, indices, threshold = min(candidates, key=lambda x: x[0])
    return [{'frame_index': i, 'video_time_ms': times[i]} for i in indices], {
        'method': 'brightness_rise_and_touch_intervals', 'threshold_gray': threshold,
        'max_residual_ms': residual, 'timing': 'approximate_flash_anchor_not_touch_time'}


def prepare_cache(session, video, touches_path, config, need_events=True):
    import mediapipe as mp

    settings = config['review']['preparation']
    tracking_settings = config['tracking']
    roi = settings['flash_roi_uv']
    if roi is not None and (len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
        raise ValueError('flash_roi_uv must be [left, top, right, bottom] in [0,1]')
    touches = pipeline.read_csv(touches_path)
    path = session / '.review_cache.json'
    identity = {'schema_version': 2, 'video_sha256': fingerprint(video),
                'touches_sha256': fingerprint(touches_path), 'mediapipe': mp.__version__,
                'tracking': tracking_settings, 'input_height_px': settings['input_height_px'],
                'continuity_implementation_sha256': fingerprint(Path(__file__).with_name('thumb_tracking.py'))}
    previous = path.read_bytes() if path.exists() else None
    cache = json.loads(previous) if previous else {}

    def save():
        nonlocal previous
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(cache, allow_nan=False), encoding='utf-8')
        if previous and not same:
            archive = session / f'.review_cache.{hashlib.sha256(previous).hexdigest()}.json'
            try:
                with archive.open('xb') as file:
                    file.write(previous)
            except FileExistsError:
                if archive.read_bytes() != previous:
                    raise ValueError(f'Cache archive differs: {archive}')
            print(f'Previous tracking preserved: {archive.name}', flush=True)
            previous = None
        os.replace(temp, path)

    same = cache.get('identity') == identity
    if cache and not same and (session / 'review.csv').exists():
        raise ValueError('Tracking inputs/config changed after review. Use the previous config '
                         'or explicitly archive review.csv and .review_cache.json before regenerating.')
    scan_tracking = not same
    def model_identity(value):
        return {**{k: value.get(k) for k in ('video_sha256', 'touches_sha256', 'mediapipe', 'input_height_px')},
                'tracking': {k: v for k, v in value.get('tracking', {}).items() if k != 'continuity'}}

    raw_saved = cache.get('raw_frames', cache.get('frames'))
    reuse_raw = (cache.get('identity', {}).get('schema_version') in (1, 2) and
                 model_identity(cache.get('identity', {})) == model_identity(identity))
    scan_brightness = not same or cache.get('flash_roi_uv') != roi
    if scan_tracking or scan_brightness:
        cap, orientation = pipeline.open_video(video)
        fps, width, height = pipeline.video_meta(cap)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if scan_tracking:
            cache = {'identity': identity, 'frames': [], 'raw_frames': [], 'tracking_status': [],
                     'image_size_px': [width, height],
                     'orientation_degrees': orientation, 'fps': fps}
        cache['brightness'] = []
        cache['flash_roi_uv'] = roi
        cache.pop('events', None)
        model = None
        try:
            if scan_tracking and not reuse_raw:
                model = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    **{k: tracking_settings[k] for k in ('max_num_hands', 'model_complexity',
                       'min_detection_confidence', 'min_tracking_confidence')})
            tracker = ThumbTracker(tracking_settings.get('continuity', {})) if scan_tracking else None
            height_small = min(height, int(settings['input_height_px']))
            width_small = round(width * height_small / height)
            if height_small <= 0:
                raise ValueError('input_height_px must be positive')
            i = 0
            print('Preparing tracking / flashes; first run only…', flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                small = cv2.resize(frame, (width_small, height_small))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    gray = gray[int(y1 * height_small):max(int(y1 * height_small)+1, int(y2 * height_small)),
                                int(x1 * width_small):max(int(x1 * width_small)+1, int(x2 * width_small))]
                cache['brightness'].append(float(gray.mean()))
                if tracker is not None:
                    time = pipeline.frame_time_ms(cap, i, fps)
                    raw = None
                    if reuse_raw:
                        raw_time, raw = raw_saved[i]
                        if abs(raw_time - time) > 0.001:
                            raise ValueError('Cached model timestamps differ from decoded frames')
                    else:
                        result = model.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                        if result.multi_hand_landmarks:
                            tip = result.multi_hand_landmarks[0].landmark[4]
                            raw = [tip.x * width, tip.y * height]
                    point, status = tracker.update(frame, raw, time)
                    cache['raw_frames'].append([time, raw])
                    cache['frames'].append([time, point])
                    cache['tracking_status'].append(status)
                i += 1
                if i % 100 == 0:
                    print(f'  {i}/{expected} frames', flush=True)
            if i != expected or len(cache['frames']) != i:
                raise ValueError(f'Video decode incomplete: {i}/{expected} frames')
        finally:
            cap.release()
            if model is not None:
                model.close()
        save()
    else:
        print(f'Reusing {path.name}', flush=True)
    if need_events:
        events, details = match_flashes([r[0] for r in cache['frames']], cache['brightness'], touches, settings)
        cache.update(events=events, alignment=details, preparation_settings=settings)
        save()
        print(f'Paired {len(events)}/{len(touches)} flashes; max interval residual '
              f'{details["max_residual_ms"]:.1f} ms (approximate synchronization)', flush=True)
    return cache
