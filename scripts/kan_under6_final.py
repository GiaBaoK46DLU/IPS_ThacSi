"""Locked, joint KAN-Mixer refit and final evaluation after validation selection.

Lives outside the frozen algorithm package. No new predictor is introduced.
Official-test labels are only read in evaluate(), after ALL refits are ready
and an immutable evaluation manifest records every checkpoint to be evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ips.data import ROOT, sha256
from ips.data_access import load_bundle
from ips.algorithms.kan.confirmation import SEEDS, immutable_json
from ips.algorithms.kan.improvement import validate_kan_only
from ips.algorithms.kan.runtime import save_json


def prepare(output):
    """Declare the primary training regime BEFORE seeing any test outcome."""
    immutable_json(output/'finalization_plan.json',{
        'confirmation_plan_sha256':sha256(output/'confirmation_plan.json'),
        'controller_sha256':sha256(Path(__file__)),
        'primary_regime':'full_development_refit','secondary_regime':'train_only',
        'primary_seeds':SEEDS,'canonical_seed_fixed_before_test':42,
        'target':'primary mean of ten per-model mean2D values strictly below6m',
        'target_threshold_m':6.,'n_test_rows_per_model':1111,'sample_std_ddof':1,
        'test_access_requires_all_refits_complete':True,
        'coordinate_metric':'unchanged original-coordinate 2D Euclidean error, all1111 rows',
        'no_test_based_regime_architecture_decoder_seed_selection':True,
        'test_evaluated_at_plan_creation':False})


def validate_layout(lock):
    if (lock.get('scope') != 'one_joint_kan_mixer'
            or lock.get('selection_split') != 'internal_validation_only'
            or lock.get('test_evaluated') is not False or lock['seeds'] != SEEDS):
        raise ValueError('Final execution requires a ten-seed, validation-only KAN-Mixer selection lock.')
    if type(lock['final_refit_epochs']) is not int or not 1 <= lock['final_refit_epochs'] <= 100:
        raise ValueError('Refit epochs must be frozen within1..100.')
    expected = {(lock['winner'],seed,'primary') for seed in SEEDS}
    if lock['winner'] != 'budget100':
        expected |= {('budget100',seed,'baseline_control') for seed in SEEDS[:3]}
    actual = [(row['name'],row['seed'],row['role']) for row in lock['sources']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Final sources must contain every prespecified seed and control, without cherry-picking.')
    if {row['sha256'] for row in lock['sources']} != set(lock['allowed_source_checkpoint_sha256']):
        raise ValueError('Allowed checkpoint hashes and declared sources disagree.')


def load_selection(output):
    plan = json.loads((output/'finalization_plan.json').read_text(encoding='utf-8'))
    if (sha256(Path(__file__)) != plan['controller_sha256']
            or sha256(output/'confirmation_plan.json') != plan['confirmation_plan_sha256']):
        raise ValueError('Final controller or confirmation plan changed after preregistration.')
    lock_path = output/'selection_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    validate_layout(lock)
    if sha256(output/'confirmation_summary.json') != lock['confirmation_summary_sha256']:
        raise ValueError('Validation confirmation summary changed after selection.')
    for name,digest in lock['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Frozen algorithm/data code changed: {name}')
    for row in lock['sources']:
        expected_path = output/'trials'/row['name']/f"seed{row['seed']}"/'calibration/pure_selected.pt'
        if Path(row['path']).resolve() != expected_path.resolve() or sha256(expected_path) != row['sha256']:
            raise ValueError('Selected checkpoint path or hash differs from validation evidence.')
    return plan,lock,lock_path


def validate_model(path, row, lock, lock_path, refitted):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    validate_kan_only(payload['config'],payload)
    if payload['seed'] != row['seed'] or payload['run_type'] != 'training':
        raise ValueError('Final model seed/type differs; smoke or arbitrary checkpoints are excluded.')
    if (payload['split_manifest_sha256'] != lock['split_manifest_sha256']
            or len(payload['classes']) != 13 or len(payload['ap_order']) != 520):
        raise ValueError('Final checkpoint must retain the frozen joint three-building UJI data schema.')
    metadata = payload['metadata']
    if refitted:
        if (not metadata.get('full_development_refit') or payload['validation_mean_2d'] is not None
                or metadata['selection_lock_sha256'] != sha256(lock_path)
                or metadata['source_checkpoint_sha256'] != row['sha256']
                or metadata['actual_counts'] != {'train':19937,'validation':0}
                or payload['epoch'] != lock['final_refit_epochs']):
            raise ValueError('Refit provenance, counts, epoch or held-out status differs from the lock.')
    elif (metadata.get('full_development_refit')
            or metadata['actual_counts'] != {'train':16025,'validation':3912}
            or payload['config']['positioning']['fit_scope'] != 'internal_train_only'):
        raise ValueError('Train-only arm must retain the original train/validation protocol.')
    return payload


def child(command, directory):
    directory.mkdir(parents=True,exist_ok=False)
    save_json(directory/'launch.json',{'command':command,'controller_sha256':sha256(Path(__file__))})
    (directory/'controller_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('RUN '+' '.join(command),flush=True)
    with (directory/'terminal.log').open('w',encoding='utf-8') as stream:
        process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError(f'Final stage failed; preserve and inspect {directory}')


def refit(output):
    _,lock,lock_path = load_selection(output)
    for row in lock['sources']:
        validate_model(Path(row['path']),row,lock,lock_path,False)
        directory = output/'final_refit'/row['name']/f"seed{row['seed']}"
        path = directory/'training/best.pt'
        if not directory.exists():
            command = [sys.executable,'-u','-m','ips.algorithms.kan.final_refit',
                       '--source-checkpoint',row['path'],'--selection-lock',str(lock_path),
                       '--run-dir',str(directory/'training'),'--seed',str(row['seed']),'--device','cuda']
            child(command,directory)
        if not (directory/'training/summary.json').is_file():
            raise ValueError(f'Incomplete refit must be inspected before a fresh retry: {directory}')
        validate_model(path,row,lock,lock_path,True)
        immutable_json(directory/'completed.json',{'checkpoint_sha256':sha256(path),
            'selection_lock_sha256':sha256(lock_path),'source_checkpoint_sha256':row['sha256'],
            'name':row['name'],'seed':row['seed'],'role':row['role'],'test_evaluated':False})
    immutable_json(output/'refit_complete.json',{'selection_lock_sha256':sha256(lock_path),
        'models':len(lock['sources']),'test_evaluated':False})


def ready_arms(output,lock,lock_path):
    """Check EVERY refit before any test labels can be read."""
    arms = []
    for row in lock['sources']:
        selected = Path(row['path'])
        fitted = output/'final_refit'/row['name']/f"seed{row['seed']}"/'training/best.pt'
        if not fitted.is_file():
            raise ValueError(f'All refits must complete before official-test evaluation: {fitted}')
        completion_path = fitted.parent.parent/'completed.json'
        if not completion_path.is_file():
            raise ValueError('All refits must have independently recorded completion before test.')
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        if (completion['checkpoint_sha256'] != sha256(fitted)
                or completion['selection_lock_sha256'] != sha256(lock_path)
                or completion['source_checkpoint_sha256'] != row['sha256']):
            raise ValueError('Refit checkpoint hash changed after its completion record.')
        validate_model(selected,row,lock,lock_path,False)
        validate_model(fitted,row,lock,lock_path,True)
        for regime,path in [('train_only',selected),('full_development_refit',fitted)]:
            arms.append({**row,'regime':regime,'checkpoint':str(path),'checkpoint_sha256':sha256(path)})
    return arms


def audit_predictions(dataset, path, metrics, expected_count=1111):
    if dataset.name != 'test' or len(dataset) != expected_count or metrics['n_samples'] != expected_count:
        raise ValueError('Final metrics must cover every prespecified official-test row.')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not np.array_equal([int(row['row_index']) for row in rows],dataset.inputs.row_indices):
        raise ValueError('Test row order/count differs; dropping difficult samples is forbidden.')
    truth = np.array([[float(row['true_x']),float(row['true_y'])] for row in rows])
    if not np.array_equal(truth,dataset.xy):
        raise ValueError('Test coordinates differ from the original frozen source.')
    prediction = np.array([[float(row['pred_x']),float(row['pred_y'])] for row in rows])
    errors = np.linalg.norm(prediction-truth,axis=1)
    saved = np.array([float(row['error_2d']) for row in rows])
    if (not np.isfinite(errors).all() or not np.allclose(errors,saved,rtol=0,atol=1e-6)
            or not math.isclose(float(errors.mean()),metrics['mean_2d'],abs_tol=1e-6,rel_tol=0)):
        raise ValueError('Reported mean must equal the unchanged 2D Euclidean error over ALL test rows.')
    return {'rows_verified':len(rows),'recomputed_mean_2d':float(errors.mean()),
            'predictions_sha256':sha256(path)}


def aggregate(rows, expected_seeds=SEEDS):
    primary = [row for row in rows if row['role']=='primary' and row['regime']=='full_development_refit']
    if sorted(row['seed'] for row in primary) != sorted(expected_seeds):
        raise ValueError('Primary result requires ALL prespecified individual-model seeds.')
    groups = {}
    for row in rows:
        groups.setdefault((row['role'],row['regime'],row['name']),[]).append(row)
    result = []
    for (role,regime,name),members in sorted(groups.items()):
        entry = {'role':role,'regime':regime,'name':name,'n_models':len(members),
                 'n_independent_test_rows_per_model':1111,'seeds':sorted(row['seed'] for row in members)}
        for key in ['mean_2d','median_2d','p90_2d','rmse_2d','building_accuracy','floor_accuracy','joint_accuracy']:
            values = [row[key] for row in members]
            entry[key+'_mean'] = statistics.mean(values)
            entry[key+'_sample_std'] = statistics.stdev(values) if len(values)>1 else None
        result.append(entry)
    main = next(row for row in result if row['role']=='primary' and row['regime']=='full_development_refit')
    return {'primary_regime':'full_development_refit','groups':result,
            'target_met_on_primary_ten_seed_mean':main['mean_2d_mean']<6.,
            'independent_test_rows':1111,'seed_runs_are_not_extra_independent_test_samples':True,
            'all_per_model_results':rows,'test_evaluated':True,
            'does_not_assert_final_report_or_goal_complete':True}


def evaluate(output):
    plan,lock,lock_path = load_selection(output)
    arms = ready_arms(output,lock,lock_path)
    manifest = {'selection_lock_sha256':sha256(lock_path),'finalization_plan_sha256':sha256(output/'finalization_plan.json'),
        'arms':arms,'primary_regime':plan['primary_regime'],'n_test_rows_per_model':1111,
        'controller_sha256':sha256(Path(__file__)),'test_evaluated_at_manifest_creation':False}
    immutable_json(output/'official_test_evaluation_manifest.json',manifest)
    # The first test split/label access occurs AFTER the immutable manifest.
    split_dir = ROOT/'data/processed/uji_group_split_v1'
    bundle = load_bundle(split_dir)
    test = bundle.split('test')
    if len(test) != 1111:
        raise ValueError('Official UJI final evaluation must use all1111 samples.')
    from ips.algorithms.kan.predict import evaluate as evaluate_checkpoint
    results = []
    for arm in arms:
        path = Path(arm['checkpoint'])
        directory = output/'official_test'/arm['regime']/arm['name']/f"seed{arm['seed']}"
        if not directory.exists():
            metrics = evaluate_checkpoint(path,split_dir,'test',directory,'cuda')
        else:
            metrics = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        if metrics['checkpoint_sha256'] != arm['checkpoint_sha256'] or metrics['split'] != 'test':
            raise ValueError('Existing test result belongs to another checkpoint or split.')
        audit = audit_predictions(test,directory/'predictions.csv',metrics)
        save_json(directory/'independent_coordinate_audit.json',audit)
        row = {key:arm[key] for key in ['name','seed','role','regime','checkpoint_sha256']}
        row.update({key:metrics[key] for key in ['mean_2d','median_2d','p90_2d','rmse_2d',
                   'building_accuracy','floor_accuracy','joint_accuracy']})
        results.append(row)
    summary = aggregate(results)
    summary['evaluation_manifest_sha256'] = sha256(output/'official_test_evaluation_manifest.json')
    immutable_json(output/'official_test_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def benchmark(output):
    _,lock,lock_path = load_selection(output)
    arms = ready_arms(output,lock,lock_path)
    # Fixed TRAINING RSSI only; no test quality data informs timings/settings.
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'data/processed/uji_group_split_v1/manifest.json').read_text(encoding='utf-8'))
    csv_path = ROOT/manifest['sources']['development']['path']
    if protocol['split_manifest_sha256'] != sha256(ROOT/'data/processed/uji_group_split_v1/manifest.json'):
        raise ValueError('Benchmark source manifest changed.')
    for arm in [row for row in arms if row['seed']==42]:
        for batch in (1,64):
            directory = output/'benchmarks'/arm['regime']/arm['name']/f'batch{batch}'
            target = directory/'benchmark.json'
            if not directory.exists():
                child([sys.executable,'-u','-m','ips','benchmark','--checkpoint',arm['checkpoint'],
                       '--csv',str(csv_path),'--output',str(target),'--batch-size',str(batch),
                       '--warmup','10','--repeats','100','--device','cuda'],directory)
            result = json.loads(target.read_text(encoding='utf-8'))
            if result['checkpoint_sha256'] != arm['checkpoint_sha256'] or result['batch_size'] != batch:
                raise ValueError('Benchmark artifact differs from the fixed model/batch.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','refit','evaluate','benchmark'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    globals()[args.stage](args.output_dir.resolve())


if __name__ == '__main__':
    main()
