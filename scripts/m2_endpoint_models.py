#!/usr/bin/env python3
"""Train causal M2 endpoint models and render predictions at centerline crossing."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import baseline_demo
import visual_pretouch as pipeline


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('m2_median_delta', 'm2_model_a_ridge', 'm2_model_b_gradient_boosting')
TITLES = {
    METHODS[0]: 'M2-0 | training median delta',
    METHODS[1]: 'M2-A | Ridge direct endpoint',
    METHODS[2]: 'M2-B | Gradient Boosting direct endpoint',
}
EXPERIMENT_NAME = 'm2_direct_endpoint_v1'


def feature_names(cfg):
    names = ['crossing_u', 'crossing_v', 'action_elapsed_ms']
    for window in cfg['history_windows_ms']:
        suffix = f'{int(window)}ms'
        names.extend([f'velocity_u_{suffix}', f'velocity_v_{suffix}',
                      f'displacement_u_{suffix}', f'displacement_v_{suffix}',
                      f'valid_points_{suffix}'])
    names.extend(['valid_ratio', 'max_valid_gap_ms', 'path_length', 'straightness',
                  'speed_change_50_to_300'])
    return names


def trial_trajectory(data, trial):
    return [row for row in data['projected_trajectory']
            if row['trial_id'] == trial['trial_id']
            and str(row['touch_index']) == str(trial['touch_index'])]


def valid_points(rows, start_ms, end_ms):
    return [row for row in rows if start_ms <= float(row['video_time_ms']) <= end_ms
            and str(row['valid']) == '1'
            and math.isfinite(float(row['thumb_u'])) and math.isfinite(float(row['thumb_v']))]


def continuous_tail(points, max_gap_ms):
    tail = []
    for row in reversed(points):
        if tail and float(tail[-1]['video_time_ms']) - float(row['video_time_ms']) > max_gap_ms:
            break
        tail.append(row)
    return list(reversed(tail))


def slope(points, min_points):
    if len(points) < min_points:
        return np.array([np.nan, np.nan])
    t = np.asarray([float(row['video_time_ms']) for row in points], dtype=np.float64)
    t -= t.mean()
    denom = t @ t
    if denom <= 0:
        return np.array([np.nan, np.nan])
    uv = np.asarray([[float(row['thumb_u']), float(row['thumb_v'])] for row in points])
    value = t @ uv / denom
    return value if np.isfinite(value).all() else np.array([np.nan, np.nan])


def sample_from_trial(data, trial, cfg, split, device_id, device_cfg):
    crossing = trial['crossing']
    reason = ('not_kept' if trial['keep'] != '1' else
              'invalid_ground_truth' if not trial['valid_gt'] else
              'no_left_to_right_crossing_before_touch' if crossing is None else '')
    sample = {k: trial[k] for k in ('session_id', 'trial_id', 'touch_index', 'keep')}
    sample.update(split=split, device_id=device_id or '', eligible=int(not reason), exclusion_reason=reason,
                  prediction_frame=crossing['frame_index'] if crossing else '',
                  prediction_time_ms=crossing['video_time_ms'] if crossing else '',
                  max_feature_time_ms=crossing['video_time_ms'] if crossing else '',
                  gt_u=trial['gt_u'], gt_v=trial['gt_v'], delta_u='', delta_v='',
                  region_width_mm=device_cfg.get('region_width_mm', ''),
                  region_height_mm=device_cfg.get('region_height_mm', ''))
    for name in feature_names(cfg):
        sample[name] = np.nan
    if crossing is None:
        return sample

    end = float(crossing['video_time_ms'])
    rows = [row for row in trial_trajectory(data, trial) if float(row['video_time_ms']) <= end]
    sample.update(crossing_u=float(crossing['thumb_u']), crossing_v=float(crossing['thumb_v']),
                  action_elapsed_ms=end - float(trial['action_start_ms']))
    velocities = {}
    for window in cfg['history_windows_ms']:
        suffix = f'{int(window)}ms'
        points = valid_points(rows, max(float(trial['action_start_ms']), end - float(window)), end)
        points = continuous_tail(points, float(data['tracking_max_gap_ms']))
        velocity = slope(points, int(cfg['min_points_per_window']))
        velocities[int(window)] = velocity
        if points:
            displacement = np.array([float(crossing['thumb_u']), float(crossing['thumb_v'])]) - np.array(
                [float(points[0]['thumb_u']), float(points[0]['thumb_v'])])
        else:
            displacement = np.array([np.nan, np.nan])
        sample.update({f'velocity_u_{suffix}': float(velocity[0]),
                       f'velocity_v_{suffix}': float(velocity[1]),
                       f'displacement_u_{suffix}': float(displacement[0]),
                       f'displacement_v_{suffix}': float(displacement[1]),
                       f'valid_points_{suffix}': len(points)})

    quality_window = float(cfg['quality_window_ms'])
    window_rows = [row for row in rows if end - quality_window <= float(row['video_time_ms']) <= end]
    points = continuous_tail(valid_points(window_rows, end - quality_window, end),
                             float(data['tracking_max_gap_ms']))
    gaps = np.diff([float(row['video_time_ms']) for row in points]) if len(points) > 1 else np.array([])
    coords = np.asarray([[float(row['thumb_u']), float(row['thumb_v'])] for row in points]) if points else np.empty((0, 2))
    path = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()) if len(coords) > 1 else np.nan
    direct = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) > 1 else np.nan
    sample.update(valid_ratio=len([row for row in window_rows if str(row['valid']) == '1']) / len(window_rows) if window_rows else np.nan,
                  max_valid_gap_ms=float(gaps.max()) if gaps.size else np.nan,
                  path_length=path,
                  straightness=direct / path if math.isfinite(path) and path > 0 else np.nan)
    short = velocities.get(50, np.array([np.nan, np.nan]))
    long = velocities.get(300, np.array([np.nan, np.nan]))
    sample['speed_change_50_to_300'] = (float(np.linalg.norm(short) - np.linalg.norm(long))
                                                 if np.isfinite(short).all() and np.isfinite(long).all() else np.nan)
    sample['delta_u'] = trial['gt_u'] - float(crossing['thumb_u'])
    sample['delta_v'] = trial['gt_v'] - float(crossing['thumb_v'])
    return sample


def prepare_samples(data, cfg, split):
    device_id, device_cfg = pipeline.evaluation_device(cfg, pipeline.read_csv(data['touches_path']))
    feature_cfg = cfg['m2']
    annotated = {**data, 'tracking_max_gap_ms': cfg['tracking']['max_gap_ms']}
    return [sample_from_trial(annotated, trial, feature_cfg, split, device_id, device_cfg)
            for trial in data['trials']]


def matrix(samples, names):
    return np.asarray([[float(sample[name]) for name in names] for sample in samples], dtype=np.float64)


def make_model(kind, params, random_state):
    if kind == 'ridge':
        regressor = TransformedTargetRegressor(regressor=Ridge(alpha=float(params['alpha'])),
                                               transformer=StandardScaler())
        steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                 ('scaler', StandardScaler()), ('regressor', regressor)]
    elif kind == 'gradient_boosting':
        base = GradientBoostingRegressor(loss=params['loss'], n_estimators=int(params['n_estimators']),
                                         learning_rate=float(params['learning_rate']),
                                         max_depth=int(params['max_depth']),
                                         min_samples_leaf=int(params['min_samples_leaf']),
                                         random_state=int(random_state))
        steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                 ('regressor', MultiOutputRegressor(base))]
    else:
        raise ValueError(f'unknown model kind: {kind}')
    return Pipeline(steps)


def candidate_parameters(kind, cfg):
    if kind == 'ridge':
        return [{'alpha': float(value)} for value in cfg['ridge_alphas']]
    values = cfg['gradient_boosting']
    keys = ('n_estimators', 'learning_rate', 'max_depth')
    return [{**{key: value for key, value in zip(keys, combination)},
             'min_samples_leaf': int(values['min_samples_leaf']), 'loss': values['loss']}
            for combination in product(*(values[key] for key in keys))]


def select_model(kind, samples, names, cfg):
    X = matrix(samples, names)
    y = np.asarray([[sample['delta_u'], sample['delta_v']] for sample in samples], dtype=np.float64)
    anchors = np.asarray([[sample['crossing_u'], sample['crossing_v']] for sample in samples])
    sizes = np.asarray([[sample['region_width_mm'], sample['region_height_mm']] for sample in samples])
    groups = np.asarray([sample['session_id'] for sample in samples])
    unique_groups = np.unique(groups)
    folds = min(int(cfg['group_cv_folds']), len(unique_groups))
    if folds < 2:
        raise ValueError('M2 model selection needs at least two training sessions')
    splitter = GroupKFold(n_splits=folds)
    results = []
    for params in candidate_parameters(kind, cfg):
        errors, fold_rows = [], []
        for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y, groups), 1):
            model = make_model(kind, params, cfg['random_state'])
            model.fit(X[train_index], y[train_index])
            raw = anchors[valid_index] + model.predict(X[valid_index])
            predicted = np.clip(raw, 0, 1)
            fold_errors = np.linalg.norm((predicted - (anchors[valid_index] + y[valid_index])) * sizes[valid_index], axis=1)
            errors.extend(fold_errors.tolist())
            fold_rows.append({'fold': fold_index,
                              'train_sessions': sorted(set(groups[train_index].tolist())),
                              'validation_sessions': sorted(set(groups[valid_index].tolist())),
                              'samples': len(valid_index),
                              'median_error_mm': float(np.median(fold_errors)),
                              'mean_error_mm': float(np.mean(fold_errors))})
        results.append({'params': params, 'median_error_mm': float(np.median(errors)),
                        'mean_error_mm': float(np.mean(errors)), 'folds': fold_rows})
    best = min(results, key=lambda item: (item['median_error_mm'], item['mean_error_mm']))
    model = make_model(kind, best['params'], cfg['random_state'])
    model.fit(X, y)
    return model, {'kind': kind, 'selection_metric': 'pooled_median_clipped_error_mm',
                   'group_key': 'session_id', 'fold_count': folds, 'candidate_results': results,
                   'selected_params': best['params'], 'training_samples': len(samples),
                   'training_sessions': sorted(unique_groups.tolist())}


def predict_sample(sample, median_delta, models, names):
    rows = []
    for method in METHODS:
        row = {key: sample[key] for key in ('session_id', 'trial_id', 'touch_index', 'keep')}
        row.update(schema_version=1, method=method, prediction_mode='centerline_crossing_direct_endpoint',
                   split=sample['split'], prediction_frame=sample['prediction_frame'],
                   prediction_time_ms=sample['prediction_time_ms'], max_feature_time_ms=sample['max_feature_time_ms'],
                   raw_pred_u='', raw_pred_v='', pred_u='', pred_v='', clipped='', predicted_delta_u='',
                   predicted_delta_v='', horizon_ms='', uses_oracle_time=0, status='failed', failure_reason='')
        if not sample['eligible']:
            row['failure_reason'] = sample['exclusion_reason']
        else:
            if method == METHODS[0]:
                delta = median_delta
            else:
                delta = models[method].predict(matrix([sample], names))[0]
            anchor = np.asarray([sample['crossing_u'], sample['crossing_v']])
            raw = anchor + np.asarray(delta)
            clipped = np.clip(raw, 0, 1)
            row.update(raw_pred_u=float(raw[0]), raw_pred_v=float(raw[1]), pred_u=float(clipped[0]),
                       pred_v=float(clipped[1]), clipped=int(np.any(raw != clipped)),
                       predicted_delta_u=float(delta[0]), predicted_delta_v=float(delta[1]), status='ok',
                       detail_line=f"predicted delta ({float(delta[0]):+.3f}, {float(delta[1]):+.3f})")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='Output directory; defaults to outputs/<version>_<timestamp>')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--training-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--last-n', type=int,
                        help='Evaluate and render only the final N test touches; default: all touches')
    parser.add_argument('--test-session', help='Override the test session for this run')
    parser.add_argument('--no-render', action='store_true', help='Save predictions and metrics without rendering videos')
    args = parser.parse_args()

    cfg = pipeline.load_yaml(args.config)
    pipeline.validate_p0_task(cfg['task'])
    split = cfg['data_split']
    configured_train_names = list(split['training_sessions'])
    configured_test_names = list(split['test_sessions'])
    if len(configured_test_names) != 1:
        raise ValueError('M2 requires exactly one session in data_split.test_sessions; target selection is automatic')
    test_name = args.test_session or configured_test_names[0]
    all_session_names = list(dict.fromkeys(configured_train_names + configured_test_names))
    if test_name not in all_session_names:
        raise ValueError(f'unknown --test-session: {test_name}')
    if args.test_session:
        train_names = [name for name in all_session_names if name != test_name]
        run_split = dict(split, training_sessions=train_names, test_sessions=[test_name])
    else:
        train_names = configured_train_names
        run_split = split
    if split.get('validation_sessions'):
        raise ValueError('M2 currently requires validation_sessions to be empty; model selection uses grouped training CV')
    if set(train_names) & {test_name}:
        raise ValueError('training_sessions and test_sessions overlap')
    if args.last_n is not None and args.last_n <= 0:
        raise ValueError('--last-n must be positive')

    target_session = (args.training_root / test_name).resolve()
    output = args.output or ROOT / 'outputs' / f'{EXPERIMENT_NAME}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

    print('Preparing reviewed training sessions ...', flush=True)
    training_data = [baseline_demo.prepare_session((args.training_root / name).resolve(), cfg) for name in train_names]
    target = baseline_demo.prepare_session(target_session, cfg)
    selected_trials = target['trials'] if args.last_n is None else target['trials'][-args.last_n:]
    if not selected_trials or (args.last_n is not None and args.last_n > len(target['trials'])):
        test_touch_count = len(target['trials'])
        raise ValueError(f'--last-n must be within the test touch count ({test_touch_count})')

    all_train_samples = [sample for data in training_data for sample in prepare_samples(data, cfg, 'train')]
    train_samples = [sample for sample in all_train_samples if sample['eligible']]
    target_samples_by_key = {(sample['trial_id'], str(sample['touch_index'])): sample
                             for sample in prepare_samples(target, cfg, 'test')}
    test_samples = [target_samples_by_key[(trial['trial_id'], str(trial['touch_index']))]
                    for trial in selected_trials]
    if not train_samples:
        raise ValueError('no eligible M2 training samples')
    names = feature_names(cfg['m2'])
    median_delta = np.median(np.asarray([[sample['delta_u'], sample['delta_v']] for sample in train_samples]), axis=0)

    print(f'Training M2 models from {len(train_samples)} samples ...', flush=True)
    ridge, ridge_selection = select_model('ridge', train_samples, names, cfg['m2'])
    boosting, boosting_selection = select_model('gradient_boosting', train_samples, names, cfg['m2'])
    models = {METHODS[1]: ridge, METHODS[2]: boosting}
    rows = [row for sample in test_samples for row in predict_sample(sample, median_delta, models, names)]
    _, device_cfg = pipeline.evaluation_device(cfg, pipeline.read_csv(target['touches_path']))
    details, metrics = baseline_demo.evaluate(rows, selected_trials, device_cfg, cfg['evaluation'], 'test', METHODS)

    pipeline.ensure_new(output)
    model_dir = output / 'models'
    model_dir.mkdir()
    joblib.dump(ridge, model_dir / 'model_a_ridge.joblib')
    joblib.dump(boosting, model_dir / 'model_b_gradient_boosting.joblib')
    pipeline.write_json(model_dir / 'median_delta.json', {'method': METHODS[0], 'delta_u': float(median_delta[0]),
                                                         'delta_v': float(median_delta[1]),
                                                         'training_samples': len(train_samples)})
    pipeline.write_json(output / 'feature_schema.json', {'schema_version': 1, 'features': names,
        'target': ['touch_u - crossing_u', 'touch_v - crossing_v'],
        'causal_cutoff': 'first valid left-to-right centerline crossing',
        'missing_values': 'training-only median imputation with missing indicators'})
    pipeline.write_json(output / 'model_selection.json', {'schema_version': 1,
        'ridge': ridge_selection, 'gradient_boosting': boosting_selection})
    pipeline.write_json(output / 'metrics.json', metrics)
    sample_rows = all_train_samples + test_samples
    sample_fields = [key for key in sample_rows[0] if not key.startswith('_')]
    pipeline.write_csv(output / 'samples.csv', sample_rows, sample_fields)
    pipeline.write_csv(output / 'predictions.csv', details, list(details[0]))
    pipeline.write_csv(output / 'trajectory.csv', target['projected_trajectory'], list(target['projected_trajectory'][0]))
    pipeline.write_json(output / 'calibration.json', target['state']['calibration'])
    shutil.copy2(args.config, output / 'config.yaml')

    videos = []
    if not args.no_render:
        for method in METHODS:
            print(f'Rendering {method} ...', flush=True)
            videos.append(baseline_demo.render(target, selected_trials, details, method,
                                               output / f'demo_{method}.mp4', cfg['render'], TITLES[method]))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    patch = subprocess.run(['git', 'diff', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    (output / 'code.patch').write_text(patch, encoding='utf-8')
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    pipeline.write_json(output / 'manifest.json', {'schema_version': 1, 'experiment': EXPERIMENT_NAME,
        'git_commit': commit, 'git_dirty': bool(patch), 'command': sys.argv,
        'data_split': run_split, 'selected_test_trials': [trial['trial_id'] for trial in selected_trials],
        'training_sources': {data['session'].name: baseline_demo.source_identity(data) for data in training_data},
        'test_source': baseline_demo.source_identity(target), 'device': device_cfg, 'videos': videos,
        'python': sys.version, 'opencv': cv2.__version__, 'numpy': np.__version__})
    print(f'Saved M2 predictions and metrics to {output}' + (f' ({len(videos)} videos)' if videos else ' (no videos)'))


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(2)
