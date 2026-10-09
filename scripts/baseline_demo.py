#!/usr/bin/env python3
"""Run the three crossing baselines from reviewed sessions and render offline clips."""

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

import review_session
import visual_pretouch as pipeline


METHODS = ('b1_crossing_projection', 'b2_linear_stat_time', 'b3_linear_oracle_time')
TITLES = ('B1 | crossing projection', 'B2 | linear + training median', 'B3 | linear + ORACLE time')
ROOT = Path(__file__).resolve().parents[1]


def history_velocity(trajectory, crossing, start_ms, cfg):
    """Fit slopes in the past continuous segment; anchor predictions at the observed point."""
    end = float(crossing['video_time_ms'])
    lower = max(start_ms, end - cfg['prediction']['linear_history_ms'])
    points = [r for r in trajectory if lower <= r['video_time_ms'] <= end and r['valid']]
    segment = []
    for row in reversed(points):
        if segment and segment[-1]['video_time_ms'] - row['video_time_ms'] > cfg['tracking']['max_gap_ms']:
            break
        segment.append(row)
    if len(segment) < cfg['prediction']['linear_min_points']:
        return None
    t = np.asarray([r['video_time_ms'] for r in segment], dtype=np.float64)
    t -= t.mean()
    if t @ t <= 0:
        return None
    uv = np.asarray([[r['thumb_u'], r['thumb_v']] for r in segment])
    slope = t @ uv / (t @ t)
    return slope if np.isfinite(slope).all() else None


def project_trajectory(trajectory, calibration, start_ms, end_ms):
    H = np.asarray(calibration['homography_image_to_screen'])
    result = []
    for row in trajectory:
        if not start_ms <= row['video_time_ms'] <= end_ms:
            continue
        uv = pipeline.project(H, [[row['thumb_tip_x_px'], row['thumb_tip_y_px']]])[0] if row['valid'] else None
        valid = uv is not None and np.isfinite(uv).all()
        result.append({**row, 'thumb_u': float(uv[0]) if valid else '',
                       'thumb_v': float(uv[1]) if valid else '', 'valid': int(valid)})
    return result


def prepare_session(session, cfg):
    if not (session / 'review.csv').is_file():
        raise ValueError('missing_review')
    if not (session / 'calibration.json').is_file():
        raise ValueError('missing_calibration: open review and mark four corners first')
    if not (session / '.review_cache.json').is_file():
        raise ValueError('missing_review_cache: prepare tracking in review first')
    video, state = review_session.prepare(session, None, None, cfg)
    touches_path = review_session.one_file(session, '*_touches.csv')
    touches = pipeline.read_csv(touches_path)
    browser_times = [float(t['touch_time_browser_ms']) for t in touches]
    if not all(math.isfinite(t) for t in browser_times) or any(b <= a for a, b in zip(browser_times, browser_times[1:])):
        raise ValueError('Touch rows must follow browser timestamp order')
    trajectory = []
    for i, (time, point) in enumerate(state['tracking']):
        valid = point is not None and np.isfinite(point).all()
        trajectory.append({'frame_index': i, 'video_time_ms': time * 1000,
                           'thumb_tip_x_px': point[0] if valid else '',
                           'thumb_tip_y_px': point[1] if valid else '',
                           'valid': int(valid)})
    reviews = {(r['trial_id'], r['touch_index']): r['keep'] for r in state['review']}
    trials, projected, previous = [], [], None
    for i, touch in enumerate(touches):
        flash = state['flashes'][i] * 1000
        time = flash - cfg['alignment']['flash_after_touch_delay_ms']
        start = 0 if previous is None else previous + cfg['prediction']['reset_gap_ms']
        previous = time  # Keep full action boundaries, including rejected trials.
        effective = state['calibrations'][i]
        calibration = effective['calibration']
        source = state['review'][effective['source_index']]
        # One fixed mapping throughout this action; mapping changes cannot create a crossing.
        action = project_trajectory(trajectory, calibration, start, time + cfg['render']['demo_after_touch_ms'])
        projected.extend({**r, 'trial_id': touch['trial_id'], 'touch_index': touch['touch_index'],
                          'calibration_source_trial_id': source['trial_id']} for r in action)
        crossings = pipeline.find_crossings(action, start, time, cfg['task']['centerline_u'],
                                             cfg['prediction']['crossing_margin_u'],
                                             cfg['tracking']['max_gap_ms'])
        crossing = crossings[0] if crossings else None
        if crossing and crossing['video_time_ms'] >= time:
            crossing = None
        gt = np.array([float(touch['touch_u']), float(touch['touch_v'])])
        valid_gt = bool(np.isfinite(gt).all() and cfg['task']['centerline_u'] < gt[0] <= 1 and 0 <= gt[1] <= 1)
        trials.append({'session_id': session.name, 'trial_id': touch['trial_id'],
                       'touch_index': touch['touch_index'], 'keep': reviews[(touch['trial_id'], touch['touch_index'])],
                       'touch_time_video_ms': time, 'flash_time_video_ms': flash,
                       'gt_u': float(gt[0]), 'gt_v': float(gt[1]), 'valid_gt': valid_gt,
                       'action_start_ms': start, 'crossing': crossing,
                       'crossings': crossings, 'crossing_count': len(crossings),
                       'calibration': calibration, 'calibration_source_trial_id': source['trial_id'],
                       'calibration_source_touch_index': source['touch_index'],
                       'calibration_inherited': effective['inherited'],
                       'velocity': history_velocity(action, crossing, start, cfg) if crossing else None})
    return {'session': session, 'video': video, 'touches_path': touches_path,
            'state': state, 'trajectory': trajectory, 'projected_trajectory': projected, 'trials': trials}


def source_identity(data):
    session = data['session']
    paths = [data['touches_path'], session / 'review.csv', session / 'calibration.json', session / '.review_cache.json']
    return {'video': {'path': str(data['video']), 'sha256': data['state']['video_sha256']},
            'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def fit_time_prior(training, cfg):
    samples = []
    for data in training:
        for trial in data['trials']:
            crossing = trial['crossing']
            if trial['keep'] == '1' and trial['valid_gt'] and crossing:
                samples.append({k: trial[k] for k in ('session_id', 'trial_id', 'touch_index')} |
                               {'remaining_time_ms': trial['touch_time_video_ms'] - crossing['video_time_ms']})
    values = [s['remaining_time_ms'] for s in samples]
    return {'schema_version': 1, 'statistic': cfg['prediction']['time_statistic'],
            'status': 'ok' if values else 'failed', 'prior_horizon_ms': float(np.median(values)) if values else None,
            'sample_count': len(values), 'samples': samples,
            'distribution_ms': {name: float(fn(values)) if values else None for name, fn in
                                [('min', np.min), ('max', np.max), ('median', np.median),
                                 ('p90', lambda v: np.percentile(v, 90, method='linear'))]},
            'sources': {data['session'].name: source_identity(data) for data in training},
            'timing': 'approximate_flash_alignment', 'split': 'train'}


def predict_position(position, velocity=None, horizon_ms=0.):
    """Causal predictor: accepts observed position, historical speed and frozen prior only."""
    position = np.asarray(position)
    raw = position if velocity is None else position + velocity * horizon_ms
    if not np.isfinite(raw).all() or not math.isfinite(horizon_ms) or horizon_ms < 0:
        return None
    return raw, np.clip(raw, 0, 1)


def oracle_prediction(crossing, velocity, touch_time_ms):
    horizon = touch_time_ms - crossing['video_time_ms']
    return horizon, predict_position([crossing['thumb_u'], crossing['thumb_v']], velocity, horizon)


def predict_trial(trial, prior_ms, velocity_scale=1.0, split='train'):
    if not 0 < velocity_scale <= 1:
        raise ValueError('prediction.velocity_scale must be in (0, 1]')
    crossing, velocity = trial['crossing'], trial['velocity']
    applied_velocity = velocity * velocity_scale if velocity is not None else None
    rows = []
    for method in METHODS:
        b1, oracle = method == METHODS[0], method == METHODS[2]
        row = {k: trial[k] for k in ('session_id', 'trial_id', 'touch_index', 'keep')}
        row.update(calibration_source_trial_id=trial.get('calibration_source_trial_id', ''),
                   calibration_source_touch_index=trial.get('calibration_source_touch_index', ''),
                   calibration_inherited=trial.get('calibration_inherited', ''),
                   calibration_reference_frame=trial.get('calibration', {}).get('reference_frame_index', ''))
        row.update(schema_version=4, method=method, prediction_mode='centerline_crossing', split=split,
                   prediction_frame=crossing['frame_index'] if crossing else '',
                   prediction_time_ms=crossing['video_time_ms'] if crossing else '',
                   raw_pred_u='', raw_pred_v='', pred_u='', pred_v='', clipped='', horizon_ms='',
                   horizon_source='none' if b1 else 'oracle_touch_time' if oracle else 'training_median',
                   velocity_u_per_ms=float(velocity[0]) if velocity is not None and not b1 else '',
                   velocity_v_per_ms=float(velocity[1]) if velocity is not None and not b1 else '',
                   velocity_scale=float(velocity_scale) if not b1 else '',
                   applied_velocity_u_per_ms=float(applied_velocity[0]) if applied_velocity is not None and not b1 else '',
                   applied_velocity_v_per_ms=float(applied_velocity[1]) if applied_velocity is not None and not b1 else '',
                   uses_oracle_time=int(oracle), status='failed', failure_reason='')
        reason = ('not_kept' if trial['keep'] != '1' else 'invalid_ground_truth' if not trial['valid_gt'] else
                  'no_left_to_right_crossing_before_touch' if crossing is None else
                  'insufficient_continuous_history' if not b1 and velocity is None else
                  'no_training_time_prior' if not b1 and not oracle and prior_ms is None else '')
        if reason:
            row['failure_reason'] = reason
        else:
            # Only the explicit oracle branch receives this trial's remaining time.
            point = np.array([crossing['thumb_u'], crossing['thumb_v']])
            if oracle:
                horizon, result = oracle_prediction(crossing, applied_velocity, trial['touch_time_video_ms'])
            else:
                horizon = 0. if b1 else prior_ms
                result = predict_position(point, None if b1 else applied_velocity, horizon)
            if result is None:
                row['failure_reason'] = 'nonfinite_prediction_or_invalid_time'
            else:
                raw, clipped = result
                row.update(raw_pred_u=float(raw[0]), raw_pred_v=float(raw[1]), pred_u=float(clipped[0]),
                           pred_v=float(clipped[1]), clipped=int(np.any(clipped != raw)),
                           horizon_ms=float(horizon), status='ok')
        rows.append(row)
    return rows


def evaluate(rows, trials, device_cfg, evaluation_cfg, split='train', methods=METHODS,
             crossing_policy='first'):
    by_key = {(t['trial_id'], t['touch_index']): t for t in trials}
    details = []
    for row in rows:
        t = by_key[(row['trial_id'], row['touch_index'])]
        result = {**row, 'gt_u': t['gt_u'], 'gt_v': t['gt_v'], 'error_normalized': '',
                  'raw_error_normalized': '', 'lead_time_ms': '', 'error_mm': '', 'raw_error_mm': ''}
        if row['status'] == 'ok':
            gt = np.array([t['gt_u'], t['gt_v']])
            delta = np.array([row['pred_u'], row['pred_v']]) - gt
            raw_delta = np.array([row['raw_pred_u'], row['raw_pred_v']]) - gt
            result.update(error_normalized=float(np.linalg.norm(delta)), raw_error_normalized=float(np.linalg.norm(raw_delta)),
                          lead_time_ms=t['touch_time_video_ms'] - row['prediction_time_ms'])
            if device_cfg.get('region_width_mm') is not None and device_cfg.get('region_height_mm') is not None:
                size = [device_cfg['region_width_mm'], device_cfg['region_height_mm']]
                result.update(error_mm=float(np.linalg.norm(delta * size)), raw_error_mm=float(np.linalg.norm(raw_delta * size)))
        details.append(result)
    common = {t['trial_id'] for t in trials if all(any(r['trial_id'] == t['trial_id'] and r['method'] == m and r['status'] == 'ok'
                                                       for r in rows) for m in methods)}
    summaries = {}
    for method in methods:
        group = [r for r in details if r['method'] == method]
        good = [r for r in group if r['status'] == 'ok']
        errors = [r['error_normalized'] for r in good]
        denominator = sum(t['keep'] == '1' and t['valid_gt']
                          and (crossing_policy != 'single_only' or t.get('crossing_count', 1) == 1)
                          for t in trials)
        summaries[method] = {'trials': len(group), 'protocol_valid_kept_trials': denominator, 'successes': len(good),
                             'coverage': len(good) / denominator if denominator else None,
                             'clipped_count': sum(r['clipped'] == 1 for r in good),
                             'clipped_fraction': sum(r['clipped'] == 1 for r in good) / len(good) if good else None,
                             'failure_reasons': dict(Counter(r['failure_reason'] for r in group if r['status'] != 'ok')),
                             'common_median_error_normalized': float(np.median([r['error_normalized'] for r in good if r['trial_id'] in common])) if common else None}
        for unit in ('normalized', 'mm'):
            values = [r[f'error_{unit}'] for r in good if r[f'error_{unit}'] != '']
            for name, fn in [('mean', np.mean), ('median', np.median), ('p90', lambda v: np.percentile(v, 90, method='linear'))]:
                summaries[method][f'{name}_error_{unit}'] = float(fn(values)) if values else None
        leads = [r['lead_time_ms'] for r in good]
        summaries[method]['lead_time_ms'] = {name: float(fn(leads)) if leads else None for name, fn in
                                           [('mean', np.mean), ('median', np.median),
                                            ('p90', lambda v: np.percentile(v, 90, method='linear'))]}
        physical = device_cfg.get('region_width_mm') is not None and device_cfg.get('region_height_mm') is not None
        for radius in evaluation_cfg['hit_radii_mm']:
            values = [r['error_mm'] for r in good if r['error_mm'] != '']
            hits = sum(v <= radius for v in values)
            summaries[method][f'hit_at_{radius}mm'] = hits / len(values) if values else None
            summaries[method][f'hit_at_{radius}mm_all_kept'] = hits / denominator if denominator and physical else None
    return details, {'schema_version': 1, 'session_id': trials[0]['session_id'], 'split': split,
                     'device': device_cfg, 'timing': 'approximate_flash_alignment',
                     'common_valid_trials': sorted(common), 'methods': summaries}


def render(data, trials, rows, method, output, cfg, title=None):
    state = data['state']
    playback_rate = float(cfg.get('playback_rate', 1.0))
    if not 0 < playback_rate <= 1:
        raise ValueError('render.playback_rate must be in (0, 1]')
    cap, _ = pipeline.open_video(data['video'])
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), state['fps'] * playback_rate,
                             (state['width'], state['height']))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot write {output}')
    predictions = {(r['trial_id'], r['touch_index']): r for r in rows if r['method'] == method}
    frames_written, clips = 0, []
    overlay_cfg = cfg.get('overlay', {})
    font_scale = float(overlay_cfg.get('font_scale', 0.68))
    line_height = int(overlay_cfg.get('line_height_px', 32))
    panel_alpha = float(overlay_cfg.get('panel_alpha', 0.82))
    panel_padding = int(overlay_cfg.get('panel_padding_px', 18))
    panel_width = int(state['width'] * float(overlay_cfg.get('panel_width_fraction', 0.56)))
    panel_width = max(180, min(state['width'] - 24, panel_width))
    panel_radius = int(overlay_cfg.get('panel_radius_px', 18))
    border_outer_width = int(cfg.get('screen_border_outer_width_px', 8))
    border_width = int(cfg.get('screen_border_width_px', 3))
    centerline_dash = float(cfg.get('centerline_dash_fraction', 0.055))
    centerline_width = int(cfg.get('centerline_width_px', 2))

    def rounded_rect(image, top_left, bottom_right, radius, color, thickness=-1):
        x1, y1 = top_left
        x2, y2 = bottom_right
        radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        if thickness < 0:
            cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            for center in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                           (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
                cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        else:
            cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def draw_text(frame, text, xy, scale, color, thickness=1):
        cv2.putText(frame, text, (xy[0] + 1, xy[1] + 2), cv2.FONT_HERSHEY_DUPLEX,
                    scale, (8, 12, 18), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_DUPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def draw_info_card(frame, method_title, meta, status, detail, status_color):
        """Draw a restrained HUD card with one clear state and minimal metadata."""
        height = panel_padding * 2 + line_height * 3
        left, top = 18, 18
        right = min(state['width'] - 18, left + panel_width)
        bottom = min(state['height'] - 18, top + height)
        layer = frame.copy()
        rounded_rect(layer, (left + 4, top + 5), (right + 4, bottom + 5), panel_radius, (14, 9, 6), -1)
        rounded_rect(layer, (left, top), (right, bottom), panel_radius, (38, 27, 20), -1)
        cv2.addWeighted(layer, panel_alpha, frame, 1.0 - panel_alpha, 0, frame)
        rounded_rect(frame, (left, top), (right, bottom), panel_radius, (116, 96, 82), 1)

        text_left = left + panel_padding
        title_y = top + panel_padding + line_height - 7
        meta_y = title_y + line_height
        detail_y = meta_y + line_height
        draw_text(frame, method_title, (text_left, title_y), font_scale, (245, 247, 250), 1)
        draw_text(frame, meta, (text_left, meta_y), font_scale * 0.78, (168, 178, 193), 1)

        pill_scale = font_scale * 0.72
        (pill_w, pill_h), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, pill_scale, 1)
        pill_left = text_left
        pill_top = detail_y - pill_h - 8
        pill_right = pill_left + pill_w + 18
        pill_bottom = detail_y + 6
        rounded_rect(frame, (pill_left, pill_top), (pill_right, pill_bottom), 8, status_color, -1)
        cv2.putText(frame, status, (pill_left + 9, detail_y), cv2.FONT_HERSHEY_DUPLEX,
                    pill_scale, (255, 255, 255), 1, cv2.LINE_AA)
        detail_left = pill_right + 12
        draw_text(frame, detail, (detail_left, detail_y), font_scale * 0.78, (214, 220, 229), 1)

    def marker(frame, uv, color, actual=False):
        layer = np.zeros_like(frame)
        xy = pipeline.norm_to_image(inverse, uv)
        if actual:
            cv2.circle(layer, xy, 11, (250, 250, 250), -1, cv2.LINE_AA)
            cv2.circle(layer, xy, 8, color, -1, cv2.LINE_AA)
            cv2.circle(layer, xy, 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            radius = int(cfg['prediction_marker_radius_px'])
            cv2.circle(layer, xy, radius + 5, (16, 22, 32), -1, cv2.LINE_AA)
            cv2.circle(layer, xy, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(layer, xy, radius, color, -1, cv2.LINE_AA)
            cv2.circle(layer, xy, max(2, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)
        pixels = mask & np.any(layer, axis=2)
        frame[pixels] = layer[pixels]  # Keep the marker itself inside the screen.

    def smooth_path(points, passes=2):
        """Chaikin smoothing keeps the observed path shape without predicting new positions."""
        path = np.asarray(points, dtype=np.float32)
        if len(path) < 3:
            return path
        for _ in range(passes):
            refined = [path[0]]
            for left, right in zip(path[:-1], path[1:]):
                refined.extend((0.75 * left + 0.25 * right, 0.25 * left + 0.75 * right))
            refined.append(path[-1])
            path = np.asarray(refined, dtype=np.float32)
        return path

    def draw_trajectory(frame, first_frame, current_frame):
        points = []
        start = max(first_frame, current_frame - int(cfg['trajectory_tail']))
        for row in data['trajectory'][start:current_frame + 1]:
            if not row['valid']:
                if points:
                    break
                continue
            points.append((row['thumb_tip_x_px'], row['thumb_tip_y_px']))
        if len(points) < 2:
            return
        path = smooth_path(points)
        for index, (left, right) in enumerate(zip(path[:-1], path[1:])):
            progress = (index + 1) / max(1, len(path) - 1)
            alpha = 0.10 + 0.72 * progress
            color = (210, round(175 + 55 * progress), round(95 - 25 * progress))
            left_px = np.rint(left).astype(int)
            right_px = np.rint(right).astype(int)
            x1 = max(0, min(left_px[0], right_px[0]) - 3)
            y1 = max(0, min(left_px[1], right_px[1]) - 3)
            x2 = min(frame.shape[1], max(left_px[0], right_px[0]) + 4)
            y2 = min(frame.shape[0], max(left_px[1], right_px[1]) + 4)
            if x1 >= x2 or y1 >= y2:
                continue
            roi = frame[y1:y2, x1:x2]
            segment = roi.copy()
            cv2.line(segment, (left_px[0] - x1, left_px[1] - y1),
                     (right_px[0] - x1, right_px[1] - y1), color, 2, cv2.LINE_AA)
            cv2.addWeighted(segment, alpha, roi, 1.0 - alpha, 0, roi)

    def draw_thumb(frame, track):
        if not track['valid']:
            return
        xy = (round(track['thumb_tip_x_px']), round(track['thumb_tip_y_px']))
        glow = frame.copy()
        cv2.circle(glow, xy, 10, (185, 235, 85), -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.18, frame, 0.82, 0, frame)
        cv2.circle(frame, xy, 5, (245, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(frame, xy, 3, (175, 235, 70), -1, cv2.LINE_AA)

    def draw_screen_frame(frame, corners):
        cv2.polylines(frame, [corners], True, (16, 22, 30), border_outer_width, cv2.LINE_AA)
        cv2.polylines(frame, [corners], True, (205, 218, 224), border_width, cv2.LINE_AA)
        accent = (225, 220, 100)
        for index, corner in enumerate(corners):
            previous_corner = corners[(index - 1) % len(corners)]
            next_corner = corners[(index + 1) % len(corners)]
            toward_previous = corner + 0.13 * (previous_corner - corner)
            toward_next = corner + 0.13 * (next_corner - corner)
            cv2.line(frame, tuple(corner), tuple(np.rint(toward_previous).astype(int)), accent, 2, cv2.LINE_AA)
            cv2.line(frame, tuple(corner), tuple(np.rint(toward_next).astype(int)), accent, 2, cv2.LINE_AA)
    try:
        for number, trial in enumerate(trials, 1):
            calibration = trial.get('calibration', state['calibration'])
            inverse = np.linalg.inv(np.asarray(calibration['homography_image_to_screen']))
            corners = np.int32(calibration['image_points_px'])
            mask = np.zeros((state['height'], state['width']), np.uint8)
            cv2.fillConvexPoly(mask, corners, 1)
            mask = mask.astype(bool)
            prediction = predictions[(trial['trial_id'], trial['touch_index'])]
            begin = max(trial['action_start_ms'], trial['touch_time_video_ms'] - cfg['demo_before_touch_ms'])
            if trial['crossing']:
                begin = max(trial['action_start_ms'], min(begin, trial['crossing']['video_time_ms'] - cfg['demo_before_crossing_ms']))
            finish = min(state['duration'] * 1000, trial['touch_time_video_ms'] + cfg['demo_after_touch_ms'])
            first = next(i for i, r in enumerate(data['trajectory']) if r['video_time_ms'] >= begin)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            clip_start = frames_written
            for i in range(first, len(data['trajectory'])):
                track = data['trajectory'][i]
                if track['video_time_ms'] > finish:
                    break
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode frame {i}')
                draw_screen_frame(frame, corners)
                centerline_layer = frame.copy()
                for start in np.arange(0.0, 1.0, max(0.01, centerline_dash * 2.0)):
                    end = min(1.0, start + centerline_dash)
                    a = pipeline.norm_to_image(inverse, (.5, float(start)))
                    b = pipeline.norm_to_image(inverse, (.5, float(end)))
                    cv2.line(centerline_layer, a, b, (80, 210, 255), centerline_width, cv2.LINE_AA)
                cv2.addWeighted(centerline_layer, 0.86, frame, 0.14, 0, frame)
                draw_trajectory(frame, first, i)
                draw_thumb(frame, track)
                visible = prediction['status'] == 'ok' and i >= prediction['prediction_frame']
                method_title = title or TITLES[METHODS.index(method)]
                split_label = str(prediction.get('split', 'train')).upper()
                meta = f"{number:02d} / {len(trials):02d}    {split_label}    {trial['trial_id']}"
                if visible:
                    marker(frame, (prediction['pred_u'], prediction['pred_v']), (70, 92, 245))
                    status = 'PREDICTED'
                    status_color = (70, 92, 225)
                    detail = f"u {prediction['pred_u']:.3f}   v {prediction['pred_v']:.3f}"
                    if prediction.get('horizon_ms', '') != '':
                        detail += f"    {float(prediction['horizon_ms']):.0f} ms"
                    if prediction.get('clipped'):
                        detail += '    clipped'
                elif prediction['status'] != 'ok':
                    status = 'FAILED'
                    status_color = (70, 80, 210)
                    detail = prediction['failure_reason']
                else:
                    status = 'TRACKING'
                    status_color = (120, 135, 155)
                    detail = 'Waiting for centerline crossing'
                if track['video_time_ms'] >= trial['touch_time_video_ms']:
                    marker(frame, (trial['gt_u'], trial['gt_v']), (255, 110, 25), actual=True)
                    if visible:
                        status = 'RESULT'
                        status_color = (190, 125, 45)
                        detail = f"lead {prediction['lead_time_ms']:.0f} ms    error {prediction['error_normalized']:.3f} norm"
                draw_info_card(frame, method_title, meta, status, detail, status_color)
                writer.write(frame)
                frames_written += 1
            clips.append({'trial_id': trial['trial_id'], 'source_begin_ms': begin, 'source_end_ms': finish,
                          'output_first_frame': clip_start, 'output_frame_count': frames_written - clip_start})
    finally:
        cap.release()
        writer.release()
    return {'path': str(output), 'frames': frames_written, 'fps': state['fps'] * playback_rate,
            'playback_rate': playback_rate, 'clips': clips, 'audio': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--training-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--last-n', type=int, help='Override render.demo_last_n')
    args = parser.parse_args()
    cfg = pipeline.load_yaml(args.config)
    pipeline.validate_p0_task(cfg['task'])
    if cfg['prediction']['time_statistic'] != 'median' or not cfg['prediction']['clip_to_screen']:
        raise ValueError('This demo requires median time and screen clipping')
    velocity_scale = float(cfg['prediction']['velocity_scale'])
    if not 0 < velocity_scale <= 1:
        raise ValueError('prediction.velocity_scale must be in (0, 1]')
    target = prepare_session(args.session_dir.resolve(), cfg)
    n = args.last_n if args.last_n is not None else cfg['render']['demo_last_n']
    if not 1 <= n <= len(target['trials']):
        raise ValueError('--last-n must be within the touch count')
    training, skipped = [], []
    split = cfg.get('data_split', {})
    names = split.get('training_sessions', cfg['prediction']['training_sessions'])
    validation_names = split.get('validation_sessions', [])
    test_names = split.get('test_sessions', [])
    if validation_names:
        raise ValueError('baseline demo currently requires an empty validation_sessions list')
    if test_names and args.session_dir.name not in test_names:
        raise ValueError(f"target session {args.session_dir.name} is not in data_split.test_sessions")
    sessions = [args.training_root / s for s in names] if names else sorted(args.training_root.glob('session_*'))
    for session in sessions:
        try:
            data = target if session.resolve() == target['session'] else prepare_session(session.resolve(), cfg)
            training.append(data)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            skipped.append({'session_id': session.name, 'reason': str(error)})
    prior = fit_time_prior(training, cfg)
    prior['requested_sessions'] = [p.name for p in sessions]
    prior['skipped_sessions'] = skipped
    prior['data_split'] = {'training_sessions': names, 'validation_sessions': validation_names,
                           'test_sessions': test_names}
    selected = target['trials'][-n:]
    target_split = 'test' if target['session'].name in test_names else 'train'
    rows = [r for t in selected for r in predict_trial(t, prior['prior_horizon_ms'], velocity_scale, target_split)]
    _, device_cfg = pipeline.evaluation_device(cfg, pipeline.read_csv(target['touches_path']))
    details, metrics = evaluate(rows, selected, device_cfg, cfg['evaluation'], target_split)
    pipeline.ensure_new(args.output)
    pipeline.write_json(args.output / 'calibration.json', target['state']['calibration'])
    pipeline.write_json(args.output / 'time_prior.json', prior)
    pipeline.write_json(args.output / 'metrics.json', metrics)
    pipeline.write_csv(args.output / 'predictions.csv', details, list(details[0]))
    trajectory = target['projected_trajectory']
    pipeline.write_csv(args.output / 'trajectory.csv', trajectory, list(trajectory[0]))
    shutil.copy2(args.config, args.output / 'config.yaml')
    videos = []
    for method in METHODS:
        print(f'Rendering {method} ...', flush=True)
        videos.append(render(target, selected, details, method, args.output / f'demo_{method}.mp4', cfg['render']))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    patch = subprocess.run(['git', 'diff', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    (args.output / 'code.patch').write_text(patch, encoding='utf-8')
    shutil.copy2(Path(__file__), args.output / 'baseline_demo.py')
    pipeline.write_json(args.output / 'manifest.json', {'schema_version': 1, 'git_commit': commit,
        'git_dirty': dirty, 'source': source_identity(target), 'device': device_cfg,
        'selected_trials': [t['trial_id'] for t in selected], 'training_time_prior': 'time_prior.json',
        'data_split': prior['data_split'],
        'command': sys.argv, 'videos': videos, 'python': sys.version, 'opencv': cv2.__version__, 'numpy': np.__version__})
    print(f'Saved {n} touch clips per baseline to {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(2)
