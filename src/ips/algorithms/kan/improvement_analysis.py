"""Validation diagnostics for the joint, single KAN-Mixer research protocol."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ...data import ROOT, sha256
from ...data_access import load_bundle
from .improvement import validate_kan_only
from .predict import evaluate
from .runtime import save_json


def coordinate_support(train, validation, predictions_path):
    """Describe seen/unseen train coordinates without changing the main metric."""
    if train.name != 'train' or validation.name != 'validation':
        raise ValueError('Coordinate-support diagnostics require the original train and validation splits.')
    with Path(predictions_path).open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not np.array_equal([int(row['row_index']) for row in rows],validation.inputs.row_indices):
        raise ValueError('Validation prediction rows or their order differ from the frozen split.')
    truth = np.array([[float(row['true_x']),float(row['true_y'])] for row in rows])
    if not np.array_equal(truth,validation.xy):
        raise ValueError('Validation coordinate labels differ from the frozen source.')
    predicted = np.array([[float(row['pred_x']),float(row['pred_y'])] for row in rows])
    errors = np.linalg.norm(predicted-truth,axis=1)
    saved_errors = np.array([float(row['error_2d']) for row in rows])
    if not np.isfinite(errors).all() or not np.allclose(errors,saved_errors,rtol=0,atol=1e-6):
        raise ValueError('Saved errors do not match the predicted coordinates in metres.')
    expected_pairs = [(validation.info.classes[c]['building'],validation.info.classes[c]['floor'])
                      for c in validation.class_ids]
    saved_pairs = [(int(row['true_building']),int(row['true_floor'])) for row in rows]
    if saved_pairs != expected_pairs:
        raise ValueError('Validation building/floor labels differ from the frozen source.')
    correct = np.array([(int(row['pred_building']),int(row['pred_floor']))==target
                        for row,target in zip(rows,expected_pairs)])
    known = {tuple(row) for row in np.column_stack([train.class_ids,train.xy])}
    seen = np.array([tuple(row) in known for row in np.column_stack([validation.class_ids,validation.xy])])
    def measure(mask):
        return {'n':int(mask.sum()),'mean_2d':float(errors[mask].mean()) if mask.any() else None,
                'joint_accuracy':float(correct[mask].mean()) if mask.any() else None}
    return {'selection_split':'internal_validation_only','test_evaluated':False,
            'predictions_sha256':sha256(Path(predictions_path)),
            'overall':measure(np.ones(len(rows),dtype=bool)),
            'seen_train_coordinate':measure(seen),'unseen_train_coordinate':measure(~seen),
            'interpretation':'Supplementary diagnostics only; keep the original full-validation mean2D. '
                'Disjoint measurement groups do not imply unseen users, devices or coordinates.'}


def analyze_completed(output):
    """Evaluate completed pure checkpoints on validation, never on test."""
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    checkpoints = []
    for complete in sorted((output/'trials').glob('*/seed*/completed.json')):
        path = complete.parent/'calibration/pure_selected.pt'
        payload = torch.load(path,map_location='cpu',weights_only=True)
        validate_kan_only(payload['config'],payload)
        if payload['split_manifest_sha256'] != protocol['split_manifest_sha256']:
            raise ValueError('Completed checkpoint split hash differs from the research protocol.')
        checkpoints.append((complete.parent,path))
        del payload
    if not checkpoints:
        raise ValueError('No completed KAN-Mixer checkpoints are available.')
    bundle = load_bundle(ROOT/'data/processed/uji_group_split_v1')
    train,validation = bundle.split('train'),bundle.split('validation')
    rows = []
    for destination,path in checkpoints:
        directory = destination/'validation_evaluation_cpu'
        if not directory.exists():
            # Only this literal validation split is accessible through the tool.
            metrics = evaluate(path,ROOT/'data/processed/uji_group_split_v1','validation',directory,'cpu')
        else:
            metrics = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
            if metrics['checkpoint_sha256'] != sha256(path) or metrics['split'] != 'validation':
                raise ValueError('Existing evaluation belongs to a different checkpoint or split.')
        support = coordinate_support(train,validation,directory/'predictions.csv')
        support['checkpoint_sha256'] = sha256(path)
        save_json(destination/'validation_support.json',support)
        rows.append({'name':destination.parent.name,'seed':metrics['seed'],
                     'validation_mean_2d':support['overall']['mean_2d'],
                     'seen_coordinate_mean_2d':support['seen_train_coordinate']['mean_2d'],
                     'unseen_coordinate_mean_2d':support['unseen_train_coordinate']['mean_2d']})
        print(json.dumps(rows[-1]),flush=True)
    save_json(output/'validation_support_analysis.json',{'test_evaluated':False,'results':rows})
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    analyze_completed(parser.parse_args().output_dir.resolve())
