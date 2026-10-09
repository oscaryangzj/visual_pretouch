#!/usr/bin/env python3
"""Train tiny causal LSTM/GRU endpoint models with session-level LOSO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import baseline_demo
import m2_endpoint_models as m2
import visual_pretouch as pipeline


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('m3_lstm_tiny', 'm3_gru_tiny')
M2_METHODS = ('m2_median_delta', 'm2_model_a_ridge', 'm2_model_b_gradient_boosting')
EXPERIMENT_NAME = 'm3_sequence_models_v1'


class TinySequenceRegressor(nn.Module):
    """One recurrent layer followed by a linear endpoint head."""

    def __init__(self, kind: str, input_dim: int, hidden_size: int, static_dim: int):
        super().__init__()
        recurrent = nn.LSTM if kind == 'lstm' else nn.GRU
        self.recurrent = recurrent(input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size + static_dim, 2)

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        output = self.recurrent(sequence)
        state = output[1]
        hidden = state[0][-1] if isinstance(self.recurrent, nn.LSTM) else state[-1]
        return self.head(torch.cat((hidden, static), dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def key_for(sample: dict) -> tuple[str, str, str]:
    return str(sample['session_id']), str(sample['trial_id']), str(sample['touch_index'])


def valid_trajectory_points(data: dict, trial: dict) -> list[dict]:
    rows = m2.trial_trajectory(data, trial)
    points = m2.valid_points(rows, float(trial['action_start_ms']), float(trial['crossing']['video_time_ms']))
    return m2.continuous_tail(points, float(data['tracking_max_gap_ms']))


def interpolate_point(points: list[dict], target_ms: float, max_gap_ms: float) -> np.ndarray | None:
    """Interpolate only inside one observed, sufficiently short valid interval."""
    if not points:
        return None
    target_ms = float(target_ms)
    times = [float(row['video_time_ms']) for row in points]
    if target_ms < times[0] - 1e-6 or target_ms > times[-1] + 1e-6:
        return None
    for row in points:
        if abs(float(row['video_time_ms']) - target_ms) <= 1e-6:
            return np.asarray([float(row['thumb_u']), float(row['thumb_v'])], dtype=np.float64)
    for left, right in zip(points, points[1:]):
        t0, t1 = float(left['video_time_ms']), float(right['video_time_ms'])
        if t0 <= target_ms <= t1 and t1 - t0 <= max_gap_ms:
            alpha = (target_ms - t0) / (t1 - t0)
            p0 = np.asarray([float(left['thumb_u']), float(left['thumb_v'])], dtype=np.float64)
            p1 = np.asarray([float(right['thumb_u']), float(right['thumb_v'])], dtype=np.float64)
            return p0 + alpha * (p1 - p0)
    return None


def build_sequence(data: dict, trial: dict, sample: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a causal fixed grid ending at the selected crossing frame."""
    crossing = trial['crossing']
    point = np.asarray([float(crossing['thumb_u']), float(crossing['thumb_v'])], dtype=np.float64)
    end_ms = float(crossing['video_time_ms'])
    window_ms = float(cfg['sequence_window_ms'])
    count = int(cfg['sequence_points'])
    grid = np.linspace(end_ms - window_ms, end_ms, count)
    points = valid_trajectory_points(data, trial)
    sequence = np.zeros((count, 3), dtype=np.float32)
    for index, target_ms in enumerate(grid):
        current = interpolate_point(points, target_ms, float(data['tracking_max_gap_ms']))
        if current is not None:
            sequence[index, :2] = (current - point).astype(np.float32)
            sequence[index, 2] = 1.0
    static = np.asarray([sample['crossing_u'], sample['crossing_v'], sample['action_elapsed_ms']], dtype=np.float32)
    return sequence, sequence[:, 2].copy(), static


def prepare_session_data(session: Path, cfg: dict, split: str) -> tuple[dict, list[dict], dict]:
    data = baseline_demo.prepare_session(session, cfg)
    data['tracking_max_gap_ms'] = cfg['tracking']['max_gap_ms']
    samples = m2.prepare_samples(data, cfg, split)
    trials = {(str(t['trial_id']), str(t['touch_index'])): t for t in data['trials']}
    sequences = {}
    for sample in samples:
        if sample['eligible']:
            trial = trials[(str(sample['trial_id']), str(sample['touch_index']))]
            sequences[key_for(sample)] = build_sequence(data, trial, sample, cfg['m3'])
    return data, samples, sequences


def fit_scaler(keys: list[tuple[str, str, str]], sequences: dict) -> dict:
    coords = []
    statics = []
    for key in keys:
        sequence, mask, static = sequences[key]
        coords.extend(sequence[mask > 0.5, :2].tolist())
        statics.append(static.tolist())
    coord_array = np.asarray(coords, dtype=np.float64)
    static_array = np.asarray(statics, dtype=np.float64)
    coord_mean = coord_array.mean(axis=0)
    coord_std = coord_array.std(axis=0)
    static_mean = static_array.mean(axis=0)
    static_std = static_array.std(axis=0)
    coord_std[coord_std < 1e-8] = 1.0
    static_std[static_std < 1e-8] = 1.0
    return {'coord_mean': coord_mean.tolist(), 'coord_std': coord_std.tolist(),
            'static_mean': static_mean.tolist(), 'static_std': static_std.tolist()}


def transform(keys: list[tuple[str, str, str]], sequences: dict, scaler: dict):
    coord_mean = np.asarray(scaler['coord_mean'], dtype=np.float32)
    coord_std = np.asarray(scaler['coord_std'], dtype=np.float32)
    static_mean = np.asarray(scaler['static_mean'], dtype=np.float32)
    static_std = np.asarray(scaler['static_std'], dtype=np.float32)
    sequence_values, static_values, masks = [], [], []
    for key in keys:
        sequence, mask, static = sequences[key]
        current = sequence.copy()
        valid = mask > 0.5
        current[:, :2] = (current[:, :2] - coord_mean) / coord_std
        current[~valid, :2] = 0.0
        current[:, 2] = mask
        sequence_values.append(current)
        static_values.append((static - static_mean) / static_std)
        masks.append(mask)
    return (np.asarray(sequence_values, dtype=np.float32),
            np.asarray(static_values, dtype=np.float32),
            np.asarray(masks, dtype=np.float32))


def labels(keys: list[tuple[str, str, str]], samples_by_key: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray([[float(samples_by_key[key]['delta_u']), float(samples_by_key[key]['delta_v'])]
                        for key in keys], dtype=np.float32)
    anchors = np.asarray([[float(samples_by_key[key]['crossing_u']), float(samples_by_key[key]['crossing_v'])]
                          for key in keys], dtype=np.float32)
    sizes = np.asarray([[float(samples_by_key[key]['region_width_mm']),
                         float(samples_by_key[key]['region_height_mm'])] for key in keys], dtype=np.float32)
    return delta, anchors, sizes


def validation_error(model, arrays, target, anchors, sizes) -> float:
    sequence, static, _ = arrays
    model.eval()
    with torch.no_grad():
        predicted_delta = model(torch.from_numpy(sequence), torch.from_numpy(static)).numpy()
    raw = anchors + predicted_delta
    predicted = np.clip(raw, 0.0, 1.0)
    errors = np.linalg.norm((predicted - (anchors + target)) * sizes, axis=1)
    return float(np.median(errors))


def train_model(kind: str, train_arrays, train_target, val_arrays, val_target, val_anchors, val_sizes,
                cfg: dict, epochs: int | None = None, early_stop: bool = True):
    mcfg = cfg['m3']
    seed = int(mcfg['random_state'])
    set_seed(seed)
    model = TinySequenceRegressor(kind, int(mcfg['input_dim']), int(mcfg['hidden_size']), int(mcfg['static_dim']))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(mcfg['learning_rate']),
                                 weight_decay=float(mcfg['weight_decay']))
    loss_fn = nn.SmoothL1Loss(beta=float(mcfg['smooth_l1_beta']))
    x, static, _ = train_arrays
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(static), torch.from_numpy(train_target))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(mcfg['batch_size']), shuffle=True, generator=generator)
    max_epochs = int(epochs or mcfg['max_epochs'])
    patience = int(mcfg['patience']) if early_stop else max_epochs
    best_state = None
    best_score = float('inf')
    best_epoch = max_epochs
    wait = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_static, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x, batch_static), batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(mcfg['gradient_clip_norm']))
            optimizer.step()
            losses.append(float(loss.detach()))
        val_score = None
        if val_arrays is not None:
            val_score = validation_error(model, val_arrays, val_target, val_anchors, val_sizes)
            if val_score < best_score:
                best_score = val_score
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_median_error_mm': val_score})
        if val_arrays is not None and wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def select_epochs(kind: str, train_keys: list[tuple[str, str, str]], sequences: dict,
                  samples_by_key: dict, session_groups: list[str], cfg: dict):
    mcfg = cfg['m3']
    unique_groups = np.unique(session_groups)
    folds = min(int(mcfg['inner_group_cv_folds']), len(unique_groups))
    if folds < 2:
        raise ValueError('M3 epoch selection needs at least two training sessions')
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(train_keys))
    best_epochs, history_rows, fold_rows = [], [], []
    for fold_index, (train_index, valid_index) in enumerate(splitter.split(dummy, dummy, session_groups), 1):
        inner_train = [train_keys[index] for index in train_index]
        inner_valid = [train_keys[index] for index in valid_index]
        scaler = fit_scaler(inner_train, sequences)
        train_arrays = transform(inner_train, sequences, scaler)
        valid_arrays = transform(inner_valid, sequences, scaler)
        train_target, _, _ = labels(inner_train, samples_by_key)
        valid_target, valid_anchors, valid_sizes = labels(inner_valid, samples_by_key)
        model, best_epoch, history = train_model(kind, train_arrays, train_target, valid_arrays,
                                                 valid_target, valid_anchors, valid_sizes, cfg)
        best_epochs.append(best_epoch)
        for row in history:
            history_rows.append({'method': f'm3_{kind}_tiny', 'phase': 'inner', 'fold': fold_index, **row})
        fold_rows.append({'fold': fold_index, 'train_sessions': sorted(set(session_groups[index] for index in train_index)),
                          'validation_sessions': sorted(set(session_groups[index] for index in valid_index)),
                          'best_epoch': best_epoch, 'samples': len(valid_index)})
    selected = max(1, int(round(float(np.median(best_epochs)))))
    return selected, history_rows, {'fold_count': folds, 'folds': fold_rows, 'best_epochs': best_epochs,
                                    'selected_epochs': selected}


def prediction_rows(samples: list[dict], predictions: dict[tuple[str, str, str], np.ndarray], method: str) -> list[dict]:
    rows = []
    for sample in samples:
        key = key_for(sample)
        row = {key_name: sample[key_name] for key_name in ('session_id', 'trial_id', 'touch_index', 'keep')}
        row.update(schema_version=1, method=method, model_family='m3', prediction_mode='centerline_crossing_sequence',
                   split=sample['split'], prediction_frame=sample['prediction_frame'],
                   prediction_time_ms=sample['prediction_time_ms'], max_feature_time_ms=sample['max_feature_time_ms'],
                   raw_pred_u='', raw_pred_v='', pred_u='', pred_v='', clipped='',
                   predicted_delta_u='', predicted_delta_v='', status='failed', failure_reason='')
        if not sample['eligible']:
            row['failure_reason'] = sample['exclusion_reason']
        else:
            delta = predictions[key]
            anchor = np.asarray([sample['crossing_u'], sample['crossing_v']], dtype=np.float64)
            raw = anchor + delta
            clipped = np.clip(raw, 0.0, 1.0)
            row.update(raw_pred_u=float(raw[0]), raw_pred_v=float(raw[1]), pred_u=float(clipped[0]),
                       pred_v=float(clipped[1]), clipped=int(np.any(raw != clipped)),
                       predicted_delta_u=float(delta[0]), predicted_delta_v=float(delta[1]), status='ok')
        rows.append(row)
    return rows


def session_metric_rows(session: str, metrics: dict) -> list[dict]:
    rows = []
    for method, values in metrics['methods'].items():
        rows.append({'session': session, 'method': method, 'trials': values['trials'],
                     'protocol_valid_kept_trials': values['protocol_valid_kept_trials'],
                     'successes': values['successes'], 'coverage': values['coverage'],
                     'mean_error_mm': values['mean_error_mm'], 'median_error_mm': values['median_error_mm'],
                     'p90_error_mm': values['p90_error_mm'],
                     'hit_at_10mm': values['hit_at_10mm'], 'hit_at_15mm': values['hit_at_15mm'],
                     'hit_at_20mm': values['hit_at_20mm'], 'hit_at_30mm': values['hit_at_30mm']})
    return rows


def macro_metrics(rows: list[dict]) -> dict:
    methods = sorted(set(row['method'] for row in rows))
    output = {}
    numeric = ('coverage', 'mean_error_mm', 'median_error_mm', 'p90_error_mm',
               'hit_at_10mm', 'hit_at_15mm', 'hit_at_20mm', 'hit_at_30mm')
    for method in methods:
        values = [row for row in rows if row['method'] == method]
        result = {'session_count': len(values)}
        for field in numeric:
            numbers = [float(row[field]) for row in values if row[field] is not None]
            result[field] = float(np.mean(numbers)) if numbers else None
        result['protocol_valid_kept_trials'] = int(sum(row['protocol_valid_kept_trials'] for row in values))
        result['successes'] = int(sum(row['successes'] for row in values))
        output[method] = result
    return output


def read_m2_reference(root: Path, session: str) -> tuple[list[dict], dict]:
    directory = root / session
    predictions_path = directory / 'predictions.csv'
    metrics_path = directory / 'metrics.json'
    if not predictions_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f'missing filtered M2 output for {session}: {directory}')
    rows = pipeline.read_csv(predictions_path)
    rows = [row for row in rows if row.get('method') in M2_METHODS]
    for row in rows:
        row['model_family'] = 'm2_reference'
    return rows, json.loads(metrics_path.read_text(encoding='utf-8'))


def source_identity(data: dict) -> dict:
    return baseline_demo.source_identity(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory; defaults to outputs/<timestamp>')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--training-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--m2-root', type=Path, default=ROOT / 'outputs/m2_single_cross_loso_v2',
                        help='Filtered M2 LOSO output used as the paired reference')
    parser.add_argument('--test-session', help='Run only one LOSO fold for a smoke test')
    args = parser.parse_args()

    cfg = pipeline.load_yaml(args.config)
    pipeline.validate_p0_task(cfg['task'])
    if cfg['prediction'].get('crossing_policy') != 'single_only':
        raise ValueError('M3 requires prediction.crossing_policy=single_only')
    split = cfg['data_split']
    sessions = list(dict.fromkeys(split['training_sessions'] + split['test_sessions']))
    if args.test_session:
        if args.test_session not in sessions:
            raise ValueError(f'unknown --test-session: {args.test_session}')
        test_sessions = [args.test_session]
    else:
        test_sessions = sessions
    m2_root = args.m2_root if args.m2_root.is_absolute() else ROOT / args.m2_root
    output = args.output or ROOT / 'outputs' / f'{EXPERIMENT_NAME}_{datetime.now().strftime("%Y%m%d%H%M%S")}'

    print('Preparing reviewed sessions and causal sequences ...', flush=True)
    prepared = {}
    all_records = []
    sequences = {}
    for session in sessions:
        split_name = 'test' if session in split['test_sessions'] else 'train'
        data, records, session_sequences = prepare_session_data((args.training_root / session).resolve(), cfg, split_name)
        prepared[session] = (data, records)
        all_records.extend(records)
        sequences.update(session_sequences)
    samples_by_key = {key_for(sample): sample for sample in all_records}
    eligible_keys = sorted(sequences)
    sequence_index = [{**sample, 'sequence_valid_points': int(np.sum(sequences[key][1] > 0.5))}
                      for sample in all_records for key in [key_for(sample)] if key in sequences]
    sequence_index.extend({**sample, 'sequence_valid_points': 0}
                          for sample in all_records if key_for(sample) not in sequences)
    sequence_index.sort(key=lambda row: (row['session_id'], row['touch_index']))

    all_fold_metrics = []
    all_prediction_rows = []
    all_history = []
    fold_records = []
    all_m2_rows = []
    model_artifacts = []
    for fold_index, test_session in enumerate(test_sessions, 1):
        train_keys = [key for key in eligible_keys if key[0] != test_session]
        test_keys = [key for key in eligible_keys if key[0] == test_session]
        if not train_keys or not test_keys:
            raise ValueError(f'empty M3 fold for {test_session}: train={len(train_keys)}, test={len(test_keys)}')
        train_groups = [key[0] for key in train_keys]
        fold_record = {'fold': fold_index, 'test_session': test_session, 'train_samples': len(train_keys),
                       'test_samples': len(test_keys), 'methods': {}}
        m2_rows, m2_metrics = read_m2_reference(m2_root, test_session)
        all_m2_rows.extend(m2_rows)
        all_fold_metrics.extend(session_metric_rows(test_session, m2_metrics))
        for method_name in METHODS:
            kind = 'lstm' if method_name.endswith('lstm_tiny') else 'gru'
            selected_epochs, history, selection = select_epochs(kind, train_keys, sequences, samples_by_key,
                                                                 train_groups, cfg)
            all_history.extend([{'fold': fold_index, 'test_session': test_session, **row} for row in history])
            scaler = fit_scaler(train_keys, sequences)
            train_arrays = transform(train_keys, sequences, scaler)
            train_target, _, _ = labels(train_keys, samples_by_key)
            model, _, final_history = train_model(kind, train_arrays, train_target, None, None, None, None,
                                                  cfg, epochs=selected_epochs, early_stop=False)
            all_history.extend([{'fold': fold_index, 'test_session': test_session, 'method': method_name,
                                 'phase': 'final', 'inner_fold': '', **row} for row in final_history])
            test_arrays = transform(test_keys, sequences, scaler)
            with torch.no_grad():
                predicted_delta = model(torch.from_numpy(test_arrays[0]), torch.from_numpy(test_arrays[1])).numpy()
            predictions = {key: value for key, value in zip(test_keys, predicted_delta)}
            target_trials = prepared[test_session][1]
            rows = prediction_rows(target_trials, predictions, method_name)
            details, metrics = baseline_demo.evaluate(rows, prepared[test_session][0]['trials'],
                                                       pipeline.evaluation_device(cfg, pipeline.read_csv(
                                                           prepared[test_session][0]['touches_path']))[1],
                                                       cfg['evaluation'], 'test', (method_name,),
                                                       crossing_policy='single_only')
            all_prediction_rows.extend(details)
            all_fold_metrics.extend(session_metric_rows(test_session, metrics))
            fold_record['methods'][method_name] = {'parameter_count': sum(p.numel() for p in model.parameters()),
                                                    'epoch_selection': selection,
                                                    'metric': metrics['methods'][method_name]}
            model_dir = output / 'models' / f'fold_{fold_index:02d}_{test_session}'
            model_artifacts.append((model_dir, method_name, model, scaler, selected_epochs, selection))
        fold_records.append(fold_record)
        print(f'[{fold_index}/{len(test_sessions)}] {test_session}: train={len(train_keys)} test={len(test_keys)}', flush=True)

    combined_predictions = all_m2_rows + all_prediction_rows
    if not combined_predictions:
        raise ValueError('no M3 predictions generated')
    pipeline.ensure_new(output)
    for model_dir, method_name, model, scaler, selected_epochs, selection in model_artifacts:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'method': method_name, 'state_dict': model.state_dict(),
                    'selected_epochs': selected_epochs, 'scaler': scaler,
                    'epoch_selection': selection}, model_dir / f'{method_name}.pt')
    eligible_sequences = [sequences[key][0] for key in eligible_keys]
    eligible_masks = [sequences[key][1] for key in eligible_keys]
    eligible_static = [sequences[key][2] for key in eligible_keys]
    np.savez_compressed(output / 'sequences.npz', sequences=np.asarray(eligible_sequences, dtype=np.float32),
                        valid=np.asarray(eligible_masks, dtype=np.float32), static=np.asarray(eligible_static, dtype=np.float32),
                        keys=np.asarray(['|'.join(key) for key in eligible_keys]))
    pipeline.write_csv(output / 'sequence_index.csv', sequence_index, list(sequence_index[0]))
    pipeline.write_csv(output / 'training_history.csv', all_history,
                       ['fold', 'test_session', 'method', 'phase', 'inner_fold', 'epoch', 'train_loss',
                        'val_median_error_mm'])
    pipeline.write_csv(output / 'predictions.csv', combined_predictions,
                       sorted(set(field for row in combined_predictions for field in row)))
    pipeline.write_csv(output / 'per_session_metrics.csv', all_fold_metrics,
                       list(all_fold_metrics[0]))
    pipeline.write_json(output / 'sequence_schema.json', {
        'schema_version': 1, 'sequence_features': ['relative_u', 'relative_v', 'valid'],
        'sequence_window_ms': float(cfg['m3']['sequence_window_ms']),
        'sequence_points': int(cfg['m3']['sequence_points']),
        'relative_time_ms': np.linspace(-float(cfg['m3']['sequence_window_ms']), 0,
                                        int(cfg['m3']['sequence_points'])).tolist(),
        'static_features': ['crossing_u', 'crossing_v', 'action_elapsed_ms'],
        'target': ['touch_u - crossing_u', 'touch_v - crossing_v'],
        'interpolation': 'causal only; between valid points with gap <= tracking.max_gap_ms',
        'crossing_policy': cfg['prediction']['crossing_policy'],
        'model_parameter_counts': {
            method: int(sum(parameter.numel() for parameter in model.parameters()))
            for _, method, model, _, _, _ in model_artifacts
        },
    })
    macro = macro_metrics(all_fold_metrics)
    pipeline.write_json(output / 'metrics.json', {'schema_version': 1, 'experiment': EXPERIMENT_NAME,
                                                  'methods': macro, 'per_session': all_fold_metrics,
                                                  'folds': fold_records,
                                                  'eligible_sequence_count': len(eligible_keys),
                                                  'all_sample_count': len(all_records)})
    shutil.copy2(args.config, output / 'config.yaml')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    patch = subprocess.run(['git', 'diff', 'HEAD'], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout
    (output / 'code.patch').write_text(patch, encoding='utf-8')
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    pipeline.write_json(output / 'manifest.json', {
        'schema_version': 1, 'experiment': EXPERIMENT_NAME, 'git_commit': commit,
        'git_dirty': bool(patch), 'command': sys.argv, 'data_split': split,
        'test_sessions': test_sessions, 'm2_reference_root': str(m2_root),
        'training_sources': {session: source_identity(prepared[session][0]) for session in sessions},
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
    })
    print(f'Saved M3 sequence predictions and metrics to {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(2)
