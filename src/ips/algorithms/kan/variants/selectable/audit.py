"""Independent native-coordinate Test audit, paired with the frozen V4 cohort."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from .....data import sha256
from .....data_access import load_bundle
from .....metrics import evaluate_predictions

METRICS=['mean_2d','median_2d','p90_2d','rmse_2d','building_accuracy','floor_accuracy','joint_accuracy']


def state_digest(checkpoint):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    digest=hashlib.sha256()
    for name,value in sorted(payload['model_state_dict'].items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_test(checkpoint,destination,split):
    from .experiment import read,immutable
    metrics=read(destination/'metrics.json')
    with (destination/'predictions.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    truth=load_bundle(split).split('test')
    indices=np.array([int(row['row_index']) for row in rows])
    xy=np.array([[float(row['pred_x']),float(row['pred_y'])] for row in rows])
    saved_truth=np.array([[float(row['true_x']),float(row['true_y'])] for row in rows])
    pairs={(c['building'],c['floor']):c['class_id'] for c in truth.info.classes}
    predicted=np.array([pairs[(int(row['pred_building']),int(row['pred_floor']))] for row in rows])
    actual=evaluate_predictions(truth.class_ids,predicted,truth.xy,xy,truth.info.classes)
    error=np.linalg.norm(xy-truth.xy,axis=1)
    saved_error=np.array([float(row['error_2d']) for row in rows])
    if (len(rows)!=1111 or not np.array_equal(indices,truth.inputs.row_indices)
        or not np.array_equal(saved_truth,truth.xy) or not np.isfinite(xy).all()
        or metrics['checkpoint_sha256']!=sha256(checkpoint)
        or not np.allclose(error,saved_error,rtol=0,atol=1e-10)
        or any(abs(actual[key]-metrics[key])>1e-10 for key in METRICS)):
        raise ValueError('Independent full Test row/truth/metric audit failed.')
    immutable(destination/'independent_audit.json',{'status':'passed','rows':1111,
        'checkpoint_sha256':sha256(checkpoint),'model_state_sha256':state_digest(checkpoint),
        'predictions_sha256':sha256(destination/'predictions.csv'),
        'metrics_recomputed':{key:actual[key] for key in METRICS}})
    return {'seed':metrics['seed'],**{key:metrics[key] for key in METRICS}}
