"""Paired, descriptive V3/V4 validation analysis. Never opens official Test data."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(output):
    protocol = read(output / 'protocol.json')
    lock = read(output / 'selection_lock.json')
    seeds = protocol['seeds']
    assert lock['seeds'] == seeds and len(seeds) == 10
    split = ROOT / 'data/processed/uji_group_split_v1'
    manifest = read(split / 'manifest.json')
    raw_path = ROOT / manifest['sources']['development']['path']
    assert digest(raw_path) == manifest['sources']['development']['sha256']
    raw = pd.read_csv(raw_path, float_precision='round_trip')
    train_index = pd.read_csv(split / 'train_indices.csv')
    val_index = pd.read_csv(split / 'validation_indices.csv')
    truth = raw.iloc[val_index.row_index.to_numpy()]
    keys = ['BUILDINGID', 'FLOOR', 'LONGITUDE', 'LATITUDE']
    known = set(raw.iloc[train_index.row_index.to_numpy()][keys].itertuples(index=False, name=None))
    seen = np.array([k in known for k in truth[keys].itertuples(index=False, name=None)])
    expected = truth[['LONGITUDE', 'LATITUDE']].to_numpy()
    provenance = {}
    errors = {}
    reporting_differences = {}
    baseline = ROOT / 'runs/kan_joint_under6_20261020_v1/trials/same_point_relative'
    for method in ('V3', 'V4'):
        rows = []
        for seed in seeds:
            path = (baseline / f'seed{seed}/validation_evaluation_cpu/predictions.csv' if method == 'V3'
                    else output / 'trials' / lock['winner'] / f'seed{seed}/validation/predictions.csv')
            frame = pd.read_csv(path, float_precision='round_trip')
            assert np.array_equal(frame.row_index.to_numpy(), val_index.row_index.to_numpy())
            assert np.array_equal(frame[['true_x', 'true_y']].to_numpy(), expected)
            e = np.linalg.norm(frame[['pred_x', 'pred_y']].to_numpy() - expected, axis=1)
            assert np.isfinite(e).all() and np.allclose(e, frame.error_2d, rtol=0, atol=1e-8)
            reported = (protocol['baseline_validation_ten']['per_seed'][str(seed)] if method == 'V3'
                        else next(r['mean_2d'] for r in lock['all_ten_validation_results'] if r['seed'] == seed))
            native_report = read(path.parent / 'metrics.json')['mean_2d']
            assert abs(float(e.mean()) - native_report) < 1e-8
            # V3 CSVs are the existing CPU replay; selection used GPU calibration.
            # Observed cross-device mean differences are below 0.000001 m.
            assert abs(float(e.mean()) - reported) < 1e-6, (method, seed, float(e.mean()), reported)
            reporting_differences[f'{method}_seed{seed}'] = float(e.mean()) - reported
            rows.append(e)
            provenance[str(path.relative_to(ROOT))] = digest(path)
        errors[method] = np.stack(rows)

    def measure(mask):
        assert mask.any()
        means = {method: float(e[:, mask].mean()) for method, e in errors.items()}
        return {'rows_per_seed': int(mask.sum()), **means, 'V4_minus_V3_m': means['V4'] - means['V3']}

    detected = (truth[[f'WAP{i:03d}' for i in range(1, 521)]].to_numpy() != 100).sum(1)
    slices = {'all': measure(np.ones(len(truth), dtype=bool)), 'seen_train_coordinate': measure(seen),
              'unseen_train_coordinate': measure(~seen), 'AP1_plus_diagnostic': measure(detected > 0)}
    for label, mask in {'AP0': detected == 0, 'AP1_to_9': (detected > 0) & (detected < 10),
                        'AP10_to_19': (detected >= 10) & (detected < 20), 'AP20_plus': detected >= 20}.items():
        if mask.any():
            slices[label] = measure(mask)
    _, groups = np.unique(val_index.group_id.to_numpy(), return_inverse=True)
    n_groups = int(groups.max()) + 1
    counts = np.bincount(groups)
    delta = errors['V4'] - errors['V3']
    sums = np.stack([np.bincount(groups, weights=e) for e in delta])
    rng = np.random.default_rng(20261026)
    conditional, two_way = [], []
    for _ in range(2000):
        sampled_groups = rng.integers(0, n_groups, n_groups)
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        denominator = counts[sampled_groups].sum()
        conditional.append(float(sums[:, sampled_groups].mean(0).sum() / denominator))
        two_way.append(float(sums[sampled_seeds][:, sampled_groups].mean(0).sum() / denominator))
    report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'paired internal Validation only',
        'seeds': seeds, 'rows_per_seed': len(truth), 'measurement_groups': n_groups,
        'slices': slices, 'seed_mean_differences_m': delta.mean(1).tolist(),
        'bootstrap': {'replicates': 2000, 'random_seed': 20261026,
            'cluster_95_percentile_interval_m': np.quantile(conditional, [.025, .975]).tolist(),
            'paired_seed_and_cluster_95_percentile_interval_m': np.quantile(two_way, [.025, .975]).tolist(),
            'caveat': 'Exploratory post-selection intervals; measurement groups and seed pairs resampled with replacement. '
                      'Not selection-adjusted, not a new gate, and not evidence about Test.'},
        'selection_lock_sha256': digest(output / 'selection_lock.json'),
        'data_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                       (raw_path, split / 'manifest.json', split / 'train_indices.csv', split / 'validation_indices.csv')},
        'CSV_minus_selection_report_m': reporting_differences,
        'device_caveat': 'Existing V3 CPU replay CSVs versus saved V4 CUDA inference; '
                         'all mean differences from frozen selection reports verified below 0.000001 m.',
        'prediction_sha256': provenance, 'analysis_script_sha256': digest(Path(__file__)),
        'test_data_opened': False, 'training_or_inference_performed': False,
        'prediction_ensemble_used': False}
    destination = output / ('paired_validation_ten_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': str(destination), 'slices': slices, 'bootstrap': report['bootstrap']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    analyze(parser.parse_args().output_dir.resolve())
