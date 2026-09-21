#!/usr/bin/env python3
"""Run M2 leave-one-session-out evaluation without rendering videos."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory containing one output directory per test session')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--training-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--last-n', type=int, help='Evaluate only the final N touches in each session')
    args = parser.parse_args()

    import yaml
    with args.config.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    sessions = list(dict.fromkeys(config['data_split']['training_sessions'] + config['data_split']['test_sessions']))
    if not sessions:
        raise ValueError('no sessions configured in data_split')
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if output_root.exists():
        raise FileExistsError(f'output directory already exists: {output_root}')
    output_root.mkdir(parents=True)

    summary = []
    for index, session in enumerate(sessions, 1):
        output = output_root / session
        command = [sys.executable, str(ROOT / 'scripts/m2_endpoint_models.py'),
                   '--config', str(args.config), '--training-root', str(args.training_root),
                   '--test-session', session, '--no-render', '--output', str(output)]
        if args.last_n is not None:
            command.extend(['--last-n', str(args.last_n)])
        print(f'[{index}/{len(sessions)}] evaluating {session}', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        for method, values in metrics['methods'].items():
            summary.append({'session': session, 'method': method,
                            'n': f"{values['successes']}/{values['trials']}",
                            'mean_mm': values['mean_error_mm'],
                            'median_mm': values['median_error_mm'],
                            'hit_at_10mm': values['hit_at_10mm'],
                            'hit_at_15mm': values['hit_at_15mm'],
                            'hit_at_20mm': values['hit_at_20mm']})

    fields = ['session', 'method', 'n', 'mean_mm', 'median_mm',
              'hit_at_10mm', 'hit_at_15mm', 'hit_at_20mm']
    with (output_root / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (output_root / 'sessions.json').write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(sessions)} session results and summary.csv to {output_root}')


if __name__ == '__main__':
    main()
