"""Independent post-run checks against frozen checkpoints and exported rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...data import ROOT, sha256
from ...data_access.supplied import read_supplied
from ...metrics import evaluate_predictions
from .supplied_experiment import load_prepared, source_settings
from .supplied_figures import csv_read
from .models.refinement import db_features
from .runtime import save_json


def verify(name,output):
    base=output/name
    completed=json.loads((base/'completed.json').read_text())
    bundle,data=load_prepared(name,output);training=data['train'];validation=data['validation']
    keys=lambda ds:{tuple([int(c),*xy]) for c,xy in zip(ds.classes.numpy(),ds.xy_original)}
    assert not keys(training)&keys(validation)
    indices_train=set(training.indices.tolist());indices_val=set(validation.indices.tolist())
    assert not indices_train&indices_val
    source=read_supplied(name,'train');test=read_supplied(name,'test')
    assert test.sources==json.loads((base/'test_source_manifest.json').read_text())['sources']
    assert np.array_equal(np.sort(np.r_[training.indices.numpy(),validation.indices.numpy()]),np.arange(len(source)))
    coord=bundle.preprocessing['coordinates']
    np.testing.assert_allclose(coord['center'],training.xy_original.mean(0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(coord['shared_scale'],max(np.sqrt(training.xy_original.var(0).mean()),1e-6),rtol=0,atol=1e-12)
    locks=json.loads((base/'pre_test_lock.json').read_text())
    asserts=[]
    for seed in completed['seeds']:
        _,decoder,settings,_=source_settings(seed)
        paths={v:base/f'seed{seed}/frozen/{v}.pt' for v in ['pure','hybrid']}
        payloads={v:torch.load(p,map_location='cpu',weights_only=True) for v,p in paths.items()}
        best=torch.load(base/f'seed{seed}/training/best.pt',map_location='cpu',weights_only=True)
        assert set(best['model_state_dict'])==set(payloads['pure']['model_state_dict'])==set(payloads['hybrid']['model_state_dict'])
        assert all(torch.equal(best['model_state_dict'][k],payloads['pure']['model_state_dict'][k]) and
                   torch.equal(best['model_state_dict'][k],payloads['hybrid']['model_state_dict'][k]) for k in best['model_state_dict'])
        for variant,payload in payloads.items():
            assert payload['seed']==seed
            assert payload['split_manifest_sha256']==sha256(bundle.directory/'manifest.json')
            assert all(payload['config']['positioning'][k]==v for k,v in decoder.items())
            lock=next(r for r in locks['checkpoints'] if r['seed']==seed and r['variant']==variant)
            assert lock['sha256']==sha256(paths[variant])
        cat=payloads['hybrid']['fingerprint_refinement']
        assert cat['settings']==settings and cat['fit_scope']=='internal_train_only'
        torch.testing.assert_close(cat['signal'],db_features(training.features),rtol=0,atol=0)
        torch.testing.assert_close(cat['xy'],training.xy,rtol=0,atol=0)
        assert torch.equal(cat['class_ids'],training.classes) and torch.equal(cat['row_indices'],training.indices)
        normalized=torch.tensor([r['xy'] for r in payloads['pure']['config']['positioning']['references']])
        labels=torch.tensor([r['class_id'] for r in payloads['pure']['config']['positioning']['references']])
        actual=set(tuple([int(c),*v]) for c,v in zip(labels,normalized.tolist()))
        expected=set(tuple([int(c),*v]) for c,v in zip(training.classes,training.xy.tolist()))
        assert actual==expected
        for split in ['train','validation','test']:
            pdir=base/f'seed{seed}/evaluation/pure/{split}';hdir=base/f'seed{seed}/evaluation/hybrid/{split}'
            rows=csv_read(pdir/'predictions.csv');hrows=csv_read(hdir/'predictions.csv')
            assert len(rows)==len(hrows)
            raw_source=test if split=='test' else source
            idx=np.arange(len(test)) if split=='test' else data[split].indices.numpy()
            assert [(r['source_file'],int(r['source_row'])) for r in rows]==list(zip(raw_source.source_files[idx].tolist(),raw_source.source_rows[idx].tolist()))
            assert all((p['source_file'],p['source_row'])==(h['source_file'],h['source_row']) for p,h in zip(rows,hrows))
            assert all(p['predicted_class']==h['predicted_class'] for p,h in zip(rows,hrows))
            for variant,vrows in [('pure',rows),('hybrid',hrows)]:
                directory=base/f'seed{seed}/evaluation/{variant}/{split}'
                true=np.asarray([[float(r['true_x']),float(r['true_y'])] for r in vrows]);predicted=np.asarray([[float(r['predicted_x']),float(r['predicted_y'])] for r in vrows])
                np.testing.assert_array_equal(true,raw_source.xy[idx])
                err=np.linalg.norm(predicted-true,axis=1)
                np.testing.assert_allclose(err,[float(r['error_2d']) for r in vrows],rtol=0,atol=1e-12)
                true_class=np.asarray([int(r['true_class']) for r in vrows]);pred_class=np.asarray([int(r['predicted_class']) for r in vrows])
                class_map={(c['building'],c['floor']):c['class_id'] for c in bundle.classes}
                np.testing.assert_array_equal(true_class,[class_map[(int(raw_source.buildings[i]),int(raw_source.floors[i]))] for i in idx])
                np.testing.assert_array_equal([int(r['true_floor']) for r in vrows],raw_source.floors[idx])
                np.testing.assert_array_equal([int(r['predicted_floor']) for r in vrows],[bundle.classes[c]['floor'] for c in pred_class])
                calculated=evaluate_predictions(true_class,pred_class,true,predicted,bundle.classes)
                saved=json.loads((directory/'metrics.json').read_text())
                for key in ['mean_2d','median_2d','p90_2d','rmse_2d','floor_accuracy','floor_macro_f1','joint_accuracy']:
                    np.testing.assert_allclose(calculated[key],saved[key],rtol=0,atol=1e-12)
                assert calculated['joint_confusion_matrix']==saved['joint_confusion_matrix']
                for key,values in raw_source.metadata.items():
                    np.testing.assert_array_equal([float(r[key]) for r in vrows],values[idx])
        asserts.append({'seed':seed,'source_decoder_and_hybrid_equal':True,'neural_weights_identical_to_best':True,
            'reference_and_fingerprint_catalogues_train_only':True,'split_manifest_and_checkpoint_hashes_verified':True,
            'prediction_alignment_all_splits':True,'metrics_recomputed_from_predictions_all_splits':True,
            'pure_hybrid_class_predictions_identical':True})
    evidence={'dataset':name,'status':'passed','seeds':asserts,'train_validation_position_overlap':0,
        'coordinate_preprocessing_fit_on_train':True,'test_used_for_model_selection':False,
        'limits':'test quality audit preceded training; training metrics include catalogue self-neighbors; DLULIB semantics are assumptions'}
    (base/'report').mkdir(exist_ok=True)
    save_json(base/'report/verification.json',evidence)
    print(f'VERIFIED {name}: {len(asserts)} seeds, all checkpoint/data/prediction/metric checks passed.',flush=True)


def main():
    import warnings
    warnings.filterwarnings('ignore',message='A single label was found',module='sklearn.metrics._classification')
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--datasets',nargs='+',default=['dlulib','hcxy']);args=p.parse_args()
    for name in args.datasets:verify(name,args.output_dir.resolve())


if __name__=='__main__':main()
