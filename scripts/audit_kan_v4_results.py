"""Audit saved KAN V4 predictions/checkpoints without any model inference.

Validation phase never parses official-test labels. Final phase requires the
completed ten-seed test summary and its pre-test checkpoint lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from ips.data import ROOT, sha256
from ips.data_access import load_bundle
from ips.algorithms.kan.research_v4 import SEEDS, verify


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, description, tolerance=1e-8):
    require(abs(float(actual)-float(expected)) <= tolerance, f'{description}: {actual} != {expected}')


def catalogue(split, preprocessing):
    keys = np.column_stack([split.class_ids, split.xy])
    unique = np.unique(keys, axis=0)
    coordinate = preprocessing['coordinates']
    xy = ((unique[:, 1:]-np.array(coordinate['center']))/coordinate['shared_scale']).astype(np.float32)
    return unique[:, 0].astype(int), xy


def weight_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_one(directory, checkpoint_path, truth, training_scope, protocol, expected_seed, refit_lock=None):
    metrics = read(directory/'metrics.json')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    checkpoint_hash = sha256(checkpoint_path)
    require(metrics['checkpoint_sha256'] == checkpoint_hash, 'Metrics/checkpoint hash mismatch.')
    require(payload['seed'] == expected_seed == metrics['seed'], 'Model seed mismatch.')
    require(payload['run_type'] == 'training', 'Smoke model is not accuracy evidence.')
    require(payload['config']['model'] == 'kan_mixer' and
            payload['config']['positioning']['variant'] == 'reference', 'Wrong architecture.')
    require(not ({'ensemble_members', 'fingerprint_refinement'} & set(payload)), 'Hybrid or ensemble excluded.')
    require(payload['ap_order'] == truth.info.ap_order and payload['classes'] == truth.info.classes and
            payload['preprocessing'] == truth.info.preprocessing, 'Input/class/coordinate mapping changed.')
    require(payload['split_manifest_sha256'] == protocol['split_manifest_sha256'], 'Split mismatch.')
    require(payload['metadata']['code_sha256'] == protocol['source_sha256'], 'Training implementation changed.')
    points = payload['config']['positioning']
    labels, positions = catalogue(training_scope, truth.info.preprocessing)
    require(len(points['references']) == len(labels), 'Catalogue size does not match fitting scope.')
    require(np.array_equal([p['class_id'] for p in points['references']], labels), 'Catalogue classes differ.')
    require(np.allclose(np.array([p['xy'] for p in points['references']]), positions, rtol=0, atol=1e-7),
            'Catalogue contains coordinates outside the declared training scope.')
    require(np.allclose(payload['model_state_dict']['reference_xy'].numpy(), positions, rtol=0, atol=1e-7),
            'Model buffers and fitted catalogue disagree.')
    require(np.array_equal(payload['model_state_dict']['reference_classes'].numpy(), labels),
            'Model reference class buffers disagree.')
    if refit_lock is not None:
        metadata = payload['metadata']
        require(metadata.get('full_development_refit') is True and
                metadata['actual_counts'] == {'train':19937, 'validation':0}, 'Refit fitting scope differs.')
        require(points['fit_scope'] == 'development_after_validation_selection', 'Refit catalogue scope differs.')
        require(payload['epoch'] == refit_lock['final_refit_epochs'], 'Refit epoch count differs.')
        require(metadata['source_checkpoint_sha256'] in refit_lock['allowed_source_checkpoint_sha256'],
                'Refit source was not validation-selected.')
        require(metadata['selection_lock_sha256'] == sha256(directory.parents[1]/'selection_lock.json'),
                'Refit was not trained under the frozen selection lock.')
        require(payload['config']['training']['sampling'] == 'reference_points', 'Selected method changed.')
    else:
        require(not payload['metadata'].get('full_development_refit', False), 'Validation is not held out after refit.')
        require(points['fit_scope'] == 'internal_train_only', 'Validation model catalogue was not train-only.')
    with (directory/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    n = len(truth.inputs)
    require(len(rows) == n, 'Prediction rows missing or duplicated.')
    indices = np.array([int(r['row_index']) for r in rows])
    require(np.array_equal(indices, truth.inputs.row_indices), 'Prediction row order differs from source.')
    require(np.array_equal([int(r['csv_line_number']) for r in rows], indices+2), 'Source CSV line mismatch.')
    actual_xy = np.array([[float(r['true_x']), float(r['true_y'])] for r in rows])
    require(np.array_equal(actual_xy, truth.xy), 'Saved truth coordinates differ from raw source.')
    pred_xy = np.array([[float(r['pred_x']), float(r['pred_y'])] for r in rows])
    require(np.isfinite(pred_xy).all(), 'Non-finite coordinates.')
    pairs = np.array([[c['building'], c['floor']] for c in truth.info.classes])
    actual_pairs = pairs[truth.class_ids]
    require(np.array_equal([[int(r['true_building']), int(r['true_floor'])] for r in rows], actual_pairs),
            'Saved truth classes differ from raw source.')
    pred_class = np.array([int(r['class_id']) for r in rows])
    require(((pred_class >= 0) & (pred_class < len(pairs))).all(), 'Invalid predicted class IDs.')
    pred_pairs = np.array([[int(r['pred_building']), int(r['pred_floor'])] for r in rows])
    require(np.array_equal(pred_pairs, pairs[pred_class]), 'Predicted class/pair mapping mismatch.')
    detected = (truth.inputs.rssi != 100).sum(1)
    require(np.array_equal([int(r['detected_aps']) for r in rows], detected), 'Input AP count mismatch.')
    errors = np.linalg.norm(pred_xy-actual_xy, axis=1)
    require(np.allclose([float(r['error_2d']) for r in rows], errors, rtol=0, atol=1e-8), 'Native Euclidean errors differ.')
    result = {'seed':expected_seed, 'n_samples':n, 'mean_2d':float(errors.mean()),
        'median_2d':float(np.median(errors)), 'p90_2d':float(np.quantile(errors, .9)),
        'rmse_2d':float(np.sqrt(np.square(errors).mean())),
        'joint_accuracy':float((pred_class == truth.class_ids).mean()),
        'building_accuracy':float((pred_pairs[:, 0] == actual_pairs[:, 0]).mean()),
        'floor_accuracy':float((pred_pairs[:, 1] == actual_pairs[:, 1]).mean())}
    for name, value in result.items():
        close(value, metrics[name], name)
    for name, target, prediction, classes in [
        ('joint',truth.class_ids,pred_class,list(range(len(pairs)))),
        ('building',actual_pairs[:, 0],pred_pairs[:, 0],sorted(set(pairs[:, 0]))),
        ('floor',actual_pairs[:, 1],pred_pairs[:, 1],sorted(set(pairs[:, 1]))),
    ]:
        matrix = confusion_matrix(target,prediction,labels=classes)
        require(np.array_equal(matrix,metrics[name+'_confusion_matrix']), f'{name} confusion differs.')
        close(f1_score(target,prediction,labels=classes,average='macro',zero_division=0),
              metrics[name+'_macro_f1'], name+' macro F1')
    result.update(checkpoint_sha256=checkpoint_hash, state_dict_sha256=weight_digest(payload['model_state_dict']),
        predictions_sha256=sha256(directory/'predictions.csv'), status='passed')
    return result


def audit(output, phase):
    protocol = verify(output)
    bundle = load_bundle(ROOT/'data/processed/uji_group_split_v1')
    train = bundle.split('train')
    results = []
    if phase == 'validation':
        truth = bundle.split('validation')
        for path in sorted(output.glob('trials/*/seed*/completed.json')):
            row = read(path)
            result = audit_one(path.parent/'validation',path.parent/'calibration/pure_selected.pt',
                               truth,train,protocol,row['seed'])
            close(result['mean_2d'],row['mean_2d'],'completed score')
            result['name'] = row['name']
            results.append(result)
        require(bool(results), 'No completed validation trials to audit.')
    else:
        # Fail before opening official-test labels unless final artifacts exist.
        summary = read(output/'official_test_summary.json')
        selection = read(output/'selection_lock.json')
        checkpoint_lock = read(output/'test_checkpoint_lock.json')
        require(summary['n_models'] == 10 and selection['seeds'] == SEEDS and
                [r['seed'] for r in checkpoint_lock['checkpoints']] == SEEDS, 'Incomplete ten-seed experiment.')
        require(checkpoint_lock['selection_lock_sha256'] == sha256(output/'selection_lock.json') and
                checkpoint_lock['test_evaluated_at_lock_creation'] is False, 'Pre-test checkpoint lock differs.')
        require([r['seed'] for r in summary['per_seed']] == SEEDS, 'Final summary seed order differs.')
        truth = bundle.split('test')
        # Catalogue scope is the original development, after selection only.
        data = bundle.development
        mapping = {(c['building'],c['floor']):c['class_id'] for c in bundle.classes}
        from types import SimpleNamespace
        development = SimpleNamespace(class_ids=np.array([mapping[p] for p in data.pairs]), xy=data.metadata[:, :2])
        for seed, locked in zip(SEEDS, checkpoint_lock['checkpoints']):
            checkpoint = output/'final_refit'/f'seed{seed}'/'best.pt'
            require(sha256(checkpoint) == locked['sha256'], 'Final model changed after pre-test lock.')
            results.append(audit_one(output/'official_test'/f'seed{seed}',checkpoint,
                                     truth,development,protocol,seed,selection))
        require(len({r['state_dict_sha256'] for r in results}) == 10, 'Seeds do not have ten distinct trained weight states.')
        for name, saved in summary['aggregate'].items():
            close(statistics.mean(r[name] for r in results), saved['mean'], name+' mean')
            close(statistics.stdev(r[name] for r in results), saved['sample_std'], name+' sample SD')
        observed = statistics.mean(r['mean_2d'] for r in results)
        require(summary['target_met'] == (observed < 7.), 'Success flag does not match ten-seed mean.')
        require(summary['no_reselection_after_test'] is True and summary['historical_test_exposure'] is True,
                'Test exposure/selection disclosure differs.')
    receipt = {'recorded_at_utc':datetime.now(timezone.utc).isoformat(), 'status':'passed',
        'phase':phase, 'models_audited':len(results), 'rows_audited':sum(r['n_samples'] for r in results),
        'test_labels_parsed':phase == 'final', 'inference_or_training_performed':False,
        'source_and_protected_baseline_hashes_verified':True, 'audit_script_sha256':sha256(Path(__file__)),
        'results':results}
    directory = output/'audits'
    directory.mkdir(exist_ok=True)
    name = phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
    (directory/name).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','phase':phase,'models':len(results),'rows':receipt['rows_audited'],
                      'receipt':str(directory/name)},ensure_ascii=False))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=['validation','final'],required=True)
    args = parser.parse_args()
    audit(args.output_dir.resolve(), args.phase)
