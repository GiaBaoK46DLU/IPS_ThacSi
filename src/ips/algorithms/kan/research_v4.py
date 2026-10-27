"""Finite validation-led KAN V4 study. No test access before a ten-seed refit lock.

python -m ips.algorithms.kan.research_v4 prepare --output-dir runs/kan_v4_20261026
python -u -m ips.algorithms.kan.research_v4 run --output-dir runs/kan_v4_20261026
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ...data import ROOT, sha256
from ...data_access import load_bundle
from .runtime import code_hashes, save_json
from .improvement_analysis import coordinate_support

SEEDS = [42, 43, 44, 100, 2024, 777, 999, 45, 46, 47]
BASE = ROOT / 'runs/kan_joint_under6_20261020_v1'
SPLIT = ROOT / 'data/processed/uji_group_split_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        save_json(path, value)


def designs():
    return {
        'ap_tokens': {'hypothesis': 'One AP per token gives channel mixing shared RSSI/mask semantics; arbitrary groups of 20 APs lose this inductive bias. Architecture package sized for one GPU.',
            'changes': {'mixer': {'tokens': 520, 'features_per_token': 2, 'channels': 16, 'token_hidden': 16, 'channel_hidden': 32}, 'positioning': {'head_width': 128}}},
        'coobservation_patches': {'hypothesis': 'Train-only AP co-observation grouping gives each 20-AP patch correlated coverage without increasing parameter count.',
            'changes': {'positioning': {'ap_grouping': 'coobservation'}}},
        'joint_spatial_risk': {'hypothesis': 'Expected anchor distance penalizes spatially distant probability mass across all building/floor classes; original XY loss supervises only the true-class branch.',
            'changes': {'loss': {'lambda_spatial_risk': 1.0}}},
        'balanced_points': {'hypothesis': 'Uniform reference-point sampling reduces the measured 2-to-80 scan imbalance per training point while retaining the original sample-weighted validation metric.',
            'changes': {'training': {'sampling': 'reference_points'}}},
        'absolute_relative_context': {'hypothesis': 'Restore absolute signal context alongside the relative RSSI invariance, while holding V3 same-point augmentation fixed. Earlier absolute-relative trial did not include the selected same-point protocol.',
            'changes': {'positioning': {'input_mode': 'absolute_relative_mask'}, 'mixer': {'features_per_token': 60}}},
    }


def prepare(output, resource_abort_study=None):
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_bundle(SPLIT)
    baseline = read(BASE / 'confirmation_summary.json')['results'][0]
    baseline3 = read(BASE / 'confirmed_architecture.json')
    ranked = next(r for r in baseline3['ranking'] if r['name'] == 'same_point_relative')
    config = read(BASE / 'configs/same_point_relative.json')
    (output / 'configs').mkdir()
    for name, design in designs().items():
        current = copy.deepcopy(config)
        for section, changes in design['changes'].items():
            current[section].update(changes)
        immutable(output / 'configs' / f'{name}.json', current)
    hashes = code_hashes()
    for rel, digest in hashes.items():
        target = output / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())
        assert sha256(target) == digest
    protected = [BASE / 'official_test_summary.json', BASE / 'selection_lock.json',
                 BASE / 'confirmation_summary.json', BASE / 'confirmed_architecture.json']
    protected += list((BASE / 'final_refit/same_point_relative').glob('seed*/training/best.pt'))
    resource_aborts = {}
    if resource_abort_study is not None:
        receipt_path = resource_abort_study.resolve()/'resource_abort.json'
        receipt = read(receipt_path)
        if (receipt.get('name') != 'ap_tokens' or receipt.get('status') != 'resource_aborted'
                or receipt.get('test_evaluated') is not False or receipt.get('epochs_observed',0)<10
                or receipt.get('observed_best_uncalibrated_validation_mean_2d',0)<100):
            raise ValueError('Resource exclusion requires the documented failed-learning AP-token probe.')
        history = Path(receipt['history_path'])
        if sha256(history) != receipt['history_sha256']:
            raise ValueError('Resource-abort history changed.')
        protected.append(receipt_path)
        resource_aborts['ap_tokens'] = receipt
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'KAN-Mixer UJI single network, 10 independent seeds',
        'base_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'branch': 'kan-version-4', 'seeds': SEEDS, 'screening_seed': 42, 'confirmation_seeds': SEEDS[:3],
        'max_screening_designs': 5, 'max_confirmed_candidates': 2,
        'stop_after_consecutive_nonimproving_validation_designs': 5,
        'minimum_validation_improvement_m': .01,
        'screening_order': list(designs()), 'designs': designs(),
        'resource_aborts': resource_aborts,
        'architecture_selection': 'lowest mean equal-weight seen/unseen validation score over seeds42/43/44, only if full-validation mean also improves V3 by0.01m',
        'refit_epoch_rule': 'rounded median of validation-selected best epochs over all10 seeds; bounded1..100',
        'primary_regime': 'full_development_refit', 'canonical_seed': 42,
        'success': 'mean of all10 per-model test mean2D strictly below7m; report sample SD; no seed selection or ensemble',
        'test_rows_per_model': 1111, 'test_evaluated': False, 'historical_test_exposure': True,
        'baseline_validation_ten': baseline, 'baseline_validation_three': ranked,
        'baseline_test_mean_2d': 7.4245239460615515, 'baseline_test_sample_std': .2935992863358274,
        'split_manifest_sha256': sha256(SPLIT / 'manifest.json'), 'source_sha256': hashes,
        'config_sha256': {p.name: sha256(p) for p in (output/'configs').glob('*.json')},
        'protected_baseline_sha256': {str(p): sha256(p) for p in protected},
        'sources': [
            {'url': 'https://arxiv.org/abs/2105.01601', 'use': 'Mixer token/channel factorization; AP token semantics is our IPS hypothesis, not a claimed paper reproduction'},
            {'url': 'https://arxiv.org/abs/2506.00656', 'use': 'AP identity and RSSI as token features; no transformer is introduced here'},
            {'url': 'https://papers.nips.cc/paper_files/paper/2015/file/a9eb812238f753132652ae09963a05e9-Paper.pdf', 'use': 'Metric-aware output distribution loss; point-target W1 reduces to expected anchor distance'}]}
    immutable(output / 'protocol.json', protocol)
    save_json(output / 'state.json', {'stage': 'prepared', 'test_evaluated': False})
    print(f'Prepared finite V4 study: {output}', flush=True)


def verify(output):
    protocol = read(output / 'protocol.json')
    if protocol['seeds'] != SEEDS or protocol['screening_order'] != list(designs()):
        raise ValueError('V4 seed list or finite hypothesis order differs from the registered study.')
    for name, expected in protocol['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Code changed after V4 registration: {name}')
    for name, expected in protocol['config_sha256'].items():
        if sha256(output/'configs'/name) != expected:
            raise ValueError(f'Design changed after registration: {name}')
    for name, expected in protocol['protected_baseline_sha256'].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'Historical baseline changed: {name}')
    if sha256(SPLIT/'manifest.json') != protocol['split_manifest_sha256']:
        raise ValueError('Split changed after registration.')
    return protocol


def trial(output, name, seed):
    verify(output)
    destination = output/'trials'/name/f'seed{seed}'
    done = destination/'completed.json'
    if done.exists():
        result = read(done)
        if (sha256(destination/'calibration/pure_selected.pt') != result['checkpoint_sha256'] or
                sha256(destination/'validation/predictions.csv') != result['validation_predictions_sha256']):
            raise ValueError('Completed trial artifacts changed.')
        return result
    if destination.exists():
        raise ValueError(f'Incomplete trial exists; preserve and audit before any recovery: {destination}')
    destination.mkdir(parents=True)
    from .train import train
    from .accuracy import calibrate
    from .predict import evaluate
    config = read(output/'configs'/f'{name}.json')
    with (destination/'terminal.log').open('w', encoding='utf-8') as log:
        from ...experiments.runner import _Tee
        import contextlib
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            train(config, destination/'training', seed, 'cuda')
            calibrate(destination/'training', destination/'calibration', 'cuda', hybrid=False)
            checkpoint = destination/'calibration/pure_selected.pt'
            metrics = evaluate(checkpoint, SPLIT, 'validation', destination/'validation', 'cuda')
    bundle = load_bundle(SPLIT)
    support = coordinate_support(bundle.split('train'), bundle.split('validation'), destination/'validation/predictions.csv')
    summary = read(destination/'training/summary.json')
    result = {'name': name, 'seed': seed, 'test_evaluated': False, 'mean_2d': metrics['mean_2d'],
        'balanced_score': (support['seen_train_coordinate']['mean_2d']+support['unseen_train_coordinate']['mean_2d'])/2,
        'support': support, 'best_epoch': summary['best_epoch'], 'training_seconds': summary['training_seconds'],
        'checkpoint_sha256': sha256(checkpoint),
        'validation_predictions_sha256': sha256(destination/'validation/predictions.csv')}
    immutable(done, result)
    print('V4_COMPLETED ' + json.dumps({k:v for k,v in result.items() if k!='support'}), flush=True)
    return result


def run(output):
    protocol = verify(output)
    if (output/'stop_report.json').exists():
        print('Study already stopped. Read stop_report.json; no automatic restart.', flush=True)
        return
    screening = []
    baseline42 = protocol['baseline_validation_ten']['per_seed']['42']
    incumbent = baseline42
    consecutive = 0
    for name in protocol['screening_order']:
        if name in protocol.get('resource_aborts', {}):
            row = {**protocol['resource_aborts'][name], 'mean_2d':None, 'balanced_score':None,
                'note':'Resource-censored trial; NOT a completed or calibrated accuracy estimate. Counts as one nonimproving design.'}
        else:
            row = trial(output, name, 42)
        screening.append(row)
        if row['mean_2d'] is not None and row['mean_2d'] < incumbent - .01:
            incumbent, consecutive = row['mean_2d'], 0
        else:
            consecutive += 1
        save_json(output/'screening.json', {'results': screening, 'incumbent_validation_mean_2d': incumbent,
            'consecutive_nonimproving': consecutive, 'test_evaluated': False})
        save_json(output/'state.json', {'stage': 'screening', 'completed_designs':len(screening), 'test_evaluated':False})
        if consecutive >= 5:
            stop(output, 'Five consecutive designs did not improve full Validation by at least0.01m.', screening)
            return
    eligible = sorted([r for r in screening if r['mean_2d'] is not None and r['mean_2d'] < baseline42-.01], key=lambda r:(r['balanced_score'], r['mean_2d'], r['name']))[:2]
    if not eligible:
        stop(output, 'Finite screening exhausted without Validation improvement.', screening)
        return
    immutable(output/'confirmation_candidates.json', {'names':[r['name'] for r in eligible], 'selection_split':'internal_validation_only','test_evaluated':False})
    confirmed = []
    for row in eligible:
        rows = [trial(output, row['name'], seed) for seed in SEEDS[:3]]
        confirmed.append({'name': row['name'], 'mean_2d': statistics.mean(r['mean_2d'] for r in rows),
            'balanced_score': statistics.mean(r['balanced_score'] for r in rows), 'rows':rows})
    baseline3 = protocol['baseline_validation_three']
    passing = [r for r in confirmed if r['mean_2d'] < baseline3['mean_full_validation_2d']-.01
        and r['balanced_score'] < baseline3['mean_balanced_selection_score']-.01]
    immutable(output/'confirmation_three.json', {'candidates': confirmed, 'test_evaluated':False})
    if not passing:
        stop(output, 'No candidate improved both full and balanced Validation over the matched three-seed V3 baseline.', confirmed)
        return
    winner = min(passing, key=lambda r:(r['balanced_score'], r['mean_2d'], r['name']))['name']
    immutable(output/'architecture_lock.json', {'winner':winner,'seeds':SEEDS,'test_evaluated':False})
    results = [trial(output, winner, seed) for seed in SEEDS]
    baseline10 = protocol['baseline_validation_ten']['mean_validation_2d']
    if statistics.mean(r['mean_2d'] for r in results) >= baseline10-.01:
        stop(output, 'Ten-seed confirmation did not improve V3 full Validation; no final test.', results)
        return
    finalize(output, winner, results)


def stop(output, reason, results):
    immutable(output/'stop_report.json', {'reason':reason,'results':results,'test_evaluated':False,
        'best_confirmed_test_result_remains_V3':7.4245239460615515,
        'next_action':'Stop computation and report; new hypothesis requires reassessment.'})
    save_json(output/'state.json', {'stage':'stopped_for_reassessment','reason':reason,'test_evaluated':False})
    print('V4_STOP ' + reason, flush=True)


def finalize(output, winner, results):
    protocol = verify(output)
    if ([r['seed'] for r in results] != SEEDS or any(r['name'] != winner for r in results)
            or any(r.get('test_evaluated') is not False for r in results)):
        raise ValueError('Finalization requires all ten validation-only winner seeds in the frozen order.')
    for row in results:
        path = output/'trials'/winner/f"seed{row['seed']}"/'calibration/pure_selected.pt'
        if sha256(path) != row['checkpoint_sha256']:
            raise ValueError('Validation-selected winner checkpoint changed before finalization.')
    epochs = max(1, min(100, round(statistics.median(r['best_epoch'] for r in results))))
    sources = [output/'trials'/winner/f'seed{s}'/'calibration/pure_selected.pt' for s in SEEDS]
    lock = {'scope':'one_joint_kan_mixer','selection_split':'internal_validation_only','test_evaluated':False,
        'winner':winner,'seeds':SEEDS,'final_refit_epochs':epochs,'source_sha256':protocol['source_sha256'],
        'split_manifest_sha256':protocol['split_manifest_sha256'],
        'allowed_source_checkpoint_sha256':[sha256(p) for p in sources],
        'training_configs_by_source_sha256':{sha256(p):{
            'path':str(p.parent.parent/'training/resolved_config.json'),
            'sha256':sha256(p.parent.parent/'training/resolved_config.json')} for p in sources},
        'all_ten_validation_results':results}
    immutable(output/'selection_lock.json', lock)
    from .final_refit import refit
    for seed, source in zip(SEEDS, sources):
        verify(output)
        destination = output/'final_refit'/f'seed{seed}'
        if not (destination/'summary.json').exists():
            with (output/f'refit_seed{seed}.log').open('w',encoding='utf-8') as log:
                import contextlib
                from ...experiments.runner import _Tee
                with contextlib.redirect_stdout(_Tee(sys.stdout,log)), contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                    refit(source, output/'selection_lock.json', destination, seed, 'cuda')
        save_json(output/'state.json', {'stage':'refit','last_completed_seed':seed,'test_evaluated':False})
    checkpoints = [output/'final_refit'/f'seed{s}'/'best.pt' for s in SEEDS]
    immutable(output/'test_checkpoint_lock.json', {'checkpoints':[{ 'seed':s,'path':str(p),'sha256':sha256(p)} for s,p in zip(SEEDS,checkpoints)],
        'selection_lock_sha256':sha256(output/'selection_lock.json'),'test_evaluated_at_lock_creation':False,
        'metric':'unchanged full1111-row original-coordinate Euclidean mean2D', 'primary_success':'ten-seed mean strictly below7m'})
    from .predict import evaluate
    test_results = []
    for seed, path in zip(SEEDS, checkpoints):
        destination = output/'official_test'/f'seed{seed}'
        if not (destination/'metrics.json').exists():
            evaluate(path,SPLIT,'test',destination,'cuda')
        metrics = read(destination/'metrics.json')
        if metrics['checkpoint_sha256'] != sha256(path):
            raise ValueError('Test checkpoint hash differs.')
        with (destination/'predictions.csv').open(encoding='utf-8',newline='') as f:
            rows = list(csv.DictReader(f))
        truth = load_bundle(SPLIT).split('test')
        indices = np.array([int(r['row_index']) for r in rows])
        xy = np.array([[float(r['pred_x']),float(r['pred_y'])] for r in rows])
        saved_truth = np.array([[float(r['true_x']),float(r['true_y'])] for r in rows])
        errors = np.linalg.norm(xy-truth.xy,axis=1)
        if (len(rows)!=1111 or not np.array_equal(indices,truth.inputs.row_indices) or
                not np.array_equal(saved_truth,truth.xy) or abs(errors.mean()-metrics['mean_2d'])>1e-10):
            raise ValueError('Independent full-test row/truth/Euclidean audit failed.')
        immutable(destination/'independent_audit.json', {'n_rows':1111,'checkpoint_sha256':sha256(path),
            'predictions_sha256':sha256(destination/'predictions.csv'),'native_mean_2d':float(errors.mean()),'status':'passed'})
        test_results.append({k:metrics[k] for k in ['seed','mean_2d','median_2d','p90_2d','rmse_2d','building_accuracy','floor_accuracy','joint_accuracy']})
    aggregate = {k:{'mean':statistics.mean(r[k] for r in test_results),'sample_std':statistics.stdev(r[k] for r in test_results)}
        for k in test_results[0] if k!='seed'}
    immutable(output/'official_test_summary.json', {'winner':winner,'primary_regime':'full_development_refit',
        'refit_epochs':epochs,'n_models':10,'test_rows_per_model':1111,'per_seed':test_results,'aggregate':aggregate,
        'target_met':aggregate['mean_2d']['mean']<7.,'test_evaluated':True,'historical_test_exposure':True,
        'no_reselection_after_test':True})
    save_json(output/'state.json', {'stage':'completed_final_evaluation','target_met':aggregate['mean_2d']['mean']<7.,'test_evaluated':True})
    print('V4_FINAL ' + json.dumps(aggregate),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--resource-abort-study',type=Path,
        help='For prepare only: carry the audited, unmodified failed-learning AP-token probe into a new protocol revision.')
    args = parser.parse_args()
    if args.action=='prepare':
        prepare(args.output_dir.resolve(), args.resource_abort_study)
    else:
        if args.resource_abort_study is not None:
            parser.error('Resource exclusions must be registered before running, through prepare.')
        run(args.output_dir.resolve())


if __name__ == '__main__':
    main()
