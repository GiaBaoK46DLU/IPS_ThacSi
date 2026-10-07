"""Summarize measured full-data GPU epochs; never invent quality or timings."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in ('mlp', 'mlp_mixer', 'kan_mixer'):
        directory = args.root / (name + '_seed42')
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        metadata = json.loads((directory / 'run_metadata.json').read_text(encoding='utf-8'))
        resources = json.loads((directory / 'resources.json').read_text(encoding='utf-8'))
        with (directory / 'history.csv').open(encoding='utf-8', newline='') as stream:
            epochs = list(csv.DictReader(stream))
        if (summary['status'] != 'completed' or len(epochs) != 3 or
                metadata['software']['device'] != 'cuda' or
                metadata['actual_counts'] != {'train': 16025, 'validation': 3912}):
            raise ValueError('Calibration requires completed CUDA runs with 3 full-data epochs.')
        measured = [float(row['epoch_seconds']) for row in epochs]
        warm = statistics.mean(measured[1:])
        rows.append({
            'model': name, 'device_name': metadata['software']['device_name'],
            'torch': metadata['software']['torch'], 'batch_size': metadata['config']['training']['batch_size'],
            'epoch_seconds_measured': measured, 'mean_epochs_2_3_seconds': warm,
            'estimated_100_epochs_seconds_per_seed': 100 * warm,
            'estimated_100_epochs_seconds_three_seeds': 300 * warm,
            'peak_allocated_mib': resources['cuda_training_phase']['peak_allocated_bytes'] / 2**20,
            'peak_reserved_mib': resources['cuda_training_phase']['peak_reserved_bytes'] / 2**20,
        })
    report = {'purpose': 'timing_and_memory_calibration_not_quality_comparison',
              'note': 'Estimate from mean of epochs 2 and 3; includes training, validation, diagnostics and checkpoint writing. Excludes startup/data loading; early stopping may shorten full runs; only one calibration seed.',
              'models': rows,
              'estimated_full_9_run_hours_without_startup': sum(row['estimated_100_epochs_seconds_three_seeds'] for row in rows) / 3600}
    (args.root / 'timing_estimate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
