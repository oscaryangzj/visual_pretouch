#!/usr/bin/env python3
"""Summarize millimetre errors and hit rates from a baseline predictions table."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


DEFAULT_RADII_MM = (10.0, 15.0, 20.0)
METHOD_LABELS = {
    'b1_crossing_projection': 'B1',
    'b2_linear_stat_time': 'B2',
    'b3_linear_oracle_time': 'B3',
    'm2_median_delta': 'M2 median delta',
    'm2_model_a_ridge': 'M2 Model A Ridge',
    'm2_model_b_gradient_boosting': 'M2 Model B Gradient Boosting',
}


def predictions_path(source):
    path = source / 'predictions.csv' if source.is_dir() else source
    if not path.is_file():
        raise FileNotFoundError(f'predictions.csv not found: {path}')
    return path


def summarize(rows, radii_mm=DEFAULT_RADII_MM):
    required = {'method', 'status', 'error_mm'}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f'predictions table must contain: {", ".join(sorted(required))}')
    methods = []
    for row in rows:
        if row['method'] not in methods:
            methods.append(row['method'])
    methods.sort(key=lambda name: (list(METHOD_LABELS).index(name) if name in METHOD_LABELS else len(METHOD_LABELS), name))
    result = []
    for method in methods:
        group = [row for row in rows if row['method'] == method]
        successful = [row for row in group if row['status'] == 'ok']
        errors = []
        for row in successful:
            value = row.get('error_mm', '')
            if value == '':
                continue
            error = float(value)
            if not math.isfinite(error) or error < 0:
                raise ValueError(f'invalid error_mm for {method}: {value}')
            errors.append(error)
        item = {
            'method': method,
            'label': METHOD_LABELS.get(method, method),
            'trials': len(group),
            'successful_predictions': len(successful),
            'mm_samples': len(errors),
            'mean_error_mm': statistics.fmean(errors) if errors else None,
            'median_error_mm': statistics.median(errors) if errors else None,
        }
        for radius in radii_mm:
            hits = sum(error <= radius for error in errors)
            key = f'hit_at_{radius:g}mm'
            item[key] = hits / len(errors) if errors else None
            item[f'{key}_count'] = hits
        result.append(item)
    return result


def print_table(summary, radii_mm):
    headers = ['method', 'n', 'mean_mm', 'median_mm'] + [f'hit@{radius:g}' for radius in radii_mm]
    rows = []
    for item in summary:
        values = [item['label'], f"{item['mm_samples']}/{item['trials']}",
                  f"{item['mean_error_mm']:.2f}" if item['mean_error_mm'] is not None else 'N/A',
                  f"{item['median_error_mm']:.2f}" if item['median_error_mm'] is not None else 'N/A']
        for radius in radii_mm:
            key = f'hit_at_{radius:g}mm'
            ratio = item[key]
            values.append('N/A' if ratio is None else
                          f"{ratio * 100:.1f}% ({item[f'{key}_count']}/{item['mm_samples']})")
        rows.append(values)
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print('  '.join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        print('  '.join(value.ljust(widths[i]) for i, value in enumerate(row)))


def print_csv(summary, radii_mm):
    fields = ['method', 'label', 'trials', 'successful_predictions', 'mm_samples',
              'mean_error_mm', 'median_error_mm']
    for radius in radii_mm:
        key = f'hit_at_{radius:g}mm'
        fields.extend([key, f'{key}_count'])
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Baseline output directory or predictions.csv')
    parser.add_argument('--radii-mm', type=float, nargs='+', default=list(DEFAULT_RADII_MM))
    parser.add_argument('--format', choices=('table', 'json', 'csv'), default='table')
    args = parser.parse_args()
    if not args.radii_mm or any(not math.isfinite(radius) or radius <= 0 for radius in args.radii_mm):
        parser.error('--radii-mm values must be positive finite numbers')
    path = predictions_path(args.source)
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'empty predictions table: {path}')
    summary = summarize(rows, args.radii_mm)
    if args.format == 'json':
        print(json.dumps({'source': str(path), 'radii_mm': args.radii_mm, 'methods': summary},
                         ensure_ascii=False, indent=2))
    elif args.format == 'csv':
        print_csv(summary, args.radii_mm)
    else:
        print_table(summary, args.radii_mm)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(2)
