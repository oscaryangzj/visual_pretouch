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
        crossing = pipeline.find_crossing(action, start, time, cfg['task']['centerline_u'],
                                          cfg['prediction']['crossing_margin_u'], cfg['tracking']['max_gap_ms'])
        if crossing and crossing['video_time_ms'] >= time:
            crossing = None
        gt = np.array([float(touch['touch_u']), float(touch['touch_v'])])
        valid_gt = bool(np.isfinite(gt).all() and cfg['task']['centerline_u'] < gt[0] <= 1 and 0 <= gt[1] <= 1)
        trials.append({'session_id': session.name, 'trial_id': touch['trial_id'],
                       'touch_index': touch['touch_index'], 'keep': reviews[(touch['trial_id'], touch['touch_index'])],
                       'touch_time_video_ms': time, 'flash_time_video_ms': flash,
                       'gt_u': float(gt[0]), 'gt_v': float(gt[1]), 'valid_gt': valid_gt,
                       'action_start_ms': start, 'crossing': crossing,
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


def evaluate(rows, trials, device_cfg, evaluation_cfg, split='train'):
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
                                                       for r in rows) for m in METHODS)}
    summaries = {}
    for method in METHODS:
        group = [r for r in details if r['method'] == method]
        good = [r for r in group if r['status'] == 'ok']
        errors = [r['error_normalized'] for r in good]
        denominator = sum(t['keep'] == '1' and t['valid_gt'] for t in trials)
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


def render(data, trials, rows, method, output, cfg):
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
    def marker(frame, uv, color, actual=False):
        layer = np.zeros_like(frame)
        xy = pipeline.norm_to_image(inverse, uv)
        if actual:
            cv2.circle(layer, xy, 13, color, 3)
        else:
            radius = int(cfg['prediction_marker_radius_px'])
            cv2.circle(layer, xy, radius + 4, (255, 255, 255), 5)
            cv2.circle(layer, xy, radius, color, 4)
            cv2.drawMarker(layer, xy, color, cv2.MARKER_CROSS, radius * 2, 4)
        pixels = mask & np.any(layer, axis=2)
        frame[pixels] = layer[pixels]  # Keep the marker itself inside the screen.
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
                cv2.polylines(frame, [corners], True, (255, 210, 50), 2)
                cv2.line(frame, pipeline.norm_to_image(inverse, (.5, 0)), pipeline.norm_to_image(inverse, (.5, 1)), (255, 50, 255), 2)
                for j in range(max(first + 1, i - cfg['trajectory_tail']), i + 1):
                    a, b = data['trajectory'][j - 1:j + 1]
                    if a['valid'] and b['valid']:
                        cv2.line(frame, (round(a['thumb_tip_x_px']), round(a['thumb_tip_y_px'])),
                                 (round(b['thumb_tip_x_px']), round(b['thumb_tip_y_px'])), (0, 220, 255), 2)
                if track['valid']:
                    cv2.circle(frame, (round(track['thumb_tip_x_px']), round(track['thumb_tip_y_px'])), 7, (0, 255, 100), 2)
                visible = prediction['status'] == 'ok' and i >= prediction['prediction_frame']
                lines = [TITLES[METHODS.index(method)] + ' | TRAINING DEMO',
                         f"{number}/{len(trials)}  {trial['trial_id']} | thumb=green, prediction=red, actual=blue after click"]
                if visible:
                    marker(frame, (prediction['pred_u'], prediction['pred_v']), (0, 70, 255))
                    lines.append(f"prediction ({prediction['pred_u']:.3f}, {prediction['pred_v']:.3f}) | horizon {prediction['horizon_ms']:.0f} ms | clipped {prediction['clipped']}")
                    if method != METHODS[0]:
                        lines.append(f"100 ms local velocity x {prediction['velocity_scale']:.2f}")
                elif prediction['status'] != 'ok':
                    lines.append('FAILED: ' + prediction['failure_reason'])
                else:
                    lines.append('Waiting for first left-to-right crossing')
                if track['video_time_ms'] >= trial['touch_time_video_ms']:
                    marker(frame, (trial['gt_u'], trial['gt_v']), (255, 150, 20), actual=True)
                    if visible:
                        lines.append(f"AFTER CLICK | lead {prediction['lead_time_ms']:.0f} ms | error {prediction['error_normalized']:.3f} normalized | approximate timing")
                for j, text in enumerate(lines):
                    xy = (20, 35 + 34 * j)
                    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 5)
                    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
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
