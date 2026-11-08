"""Frozen, resumable S-KAN/V4 comparison: ten seeds and all1111 Test rows."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import statistics
import subprocess
from datetime import datetime,timezone
from pathlib import Path

import torch

from .....data import ROOT,sha256
from .....data_access import load_bundle
from ...runtime import code_hashes,load_checkpoint,save_json
from .audit import METRICS,audit_test,state_digest

SEEDS=[42,43,44,100,2024,777,999,45,46,47]
CONFIG=ROOT/'src/ips/algorithms/kan/configs/selectable/uji_v4_matched.json'
V4_CONFIG=ROOT/'src/ips/algorithms/kan/configs/v4_balanced_points.json'
SPLIT=ROOT/'data/processed/uji_group_split_v1'
BASELINE=ROOT/'runs/kan_v4_20261026_r2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:
            raise ValueError(f'Frozen artifact changed: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        save_json(path,value)


def training_design(config):
    result=copy.deepcopy(config)
    for key in ('references','fit_scope'):
        result['positioning'].pop(key,None)
    return result


def validate_matched(config,baseline):
    result=copy.deepcopy(config)
    extra=result['kan']
    if extra.pop('backend',None)!='selectable' or extra.pop('selection_stage',None)!='search':
        raise ValueError('Expected explicit selectable search backend.')
    from .layers import validate_pool
    from .bases import default_pool
    pool=extra.pop('candidates')
    validate_pool(pool)
    if pool!=default_pool():
        raise ValueError('This protocol requires the declared eighteen-candidate IPS pool.')
    if result!=baseline:
        raise ValueError('Outer architecture, data, loss, or main training differs from V4.')


def refit_epochs(results):
    if [r['seed'] for r in results]!=SEEDS:
        raise ValueError('Exactly all ten seeds, in declared order, are required.')
    epochs=[r['best_epoch'] for r in results]
    if any(type(e) is not int or not 1<=e<=100 for e in epochs):
        raise ValueError('Invalid Validation-selected epoch.')
    return round(statistics.median(epochs))


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    config=read(CONFIG);baseline_config=read(V4_CONFIG)
    validate_matched(config,baseline_config)
    bundle=load_bundle(SPLIT)
    if [len(bundle.indices[x]) for x in ('train','validation','test')]!=[16025,3912,1111]:
        raise ValueError('Unexpected UJI split.')
    baseline=read(BASELINE/'official_test_summary.json')
    if (baseline['n_models']!=10 or baseline['test_rows_per_model']!=1111
        or [r['seed'] for r in baseline['per_seed']]!=SEEDS):
        raise ValueError('V4 baseline cohort is incomplete.')
    guards=code_hashes()
    for path in (CONFIG,V4_CONFIG,ROOT/'scripts/run_skan_uji.py',SPLIT/'manifest.json',
                 Path(__file__).parent/'provenance.json',Path(__file__).parent/'README.md'):
        guards[path.relative_to(ROOT).as_posix()]=sha256(path)
    baseline_paths=[BASELINE/name for name in ('protocol.json','selection_lock.json',
        'test_checkpoint_lock.json','official_test_summary.json')]
    for seed in SEEDS:
        baseline_paths += [BASELINE/'final_refit'/f'seed{seed}'/name
                          for name in ('best.pt','resolved_config.json','run_metadata.json','summary.json')]
        baseline_paths += [BASELINE/'official_test'/f'seed{seed}'/name
                          for name in ('metrics.json','predictions.csv')]
    baseline_paths+=list((BASELINE/'source_snapshot').rglob('*'))
    guards.update({p.relative_to(ROOT).as_posix():sha256(p) for p in baseline_paths if p.is_file()})
    output.mkdir(parents=True)
    for rel in guards:
        if rel.startswith('src/') or rel=='scripts/run_skan_uji.py':
            target=output/'source_snapshot'/rel
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    immutable(output/'protocol.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'seeds':SEEDS,'config':config,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'guard_sha256':guards,'split_manifest_sha256':sha256(SPLIT/'manifest.json'),
        'baseline':{'path':str(BASELINE),'config':baseline_config,'result':baseline},
        'search':{'fraction':.01,'batch_size':16,'full_epochs':3,'selective_epochs':3,'learning_rate':.001,
            'prune_per_node_per_round':1,'rounds':17,'selection_weights':'masked_softmax_per_input_node',
            'optimizer_reset_each_phase':True,'scheduler':None,'selection_scope':'internal_train_only'},
        'refit_epoch_rule':'round(median(all ten Validation-best epochs)); cosine horizon stays100',
        'decoder_rule':'per-seed Validation only: k=[1,3,5,10,20,1000], T=[0.5,1,2,4]',
        'counts':{'train':16025,'validation':3912,'development':19937,'test':1111},
        'test_policy':'freeze all ten refits before new Test; full1111 native2D predictions/model; no Test tuning',
        'historical_test_exposure':True,'all_ten_training_and_refit_weights_fresh':True,
        'comparison_limit':'fixed V4 outer/main-training settings; additional selection compute and different parameter counts',
        'implementation_claim':'independent adaptation of S-KAN Algorithm1; explicitly declared IPS pool, not author-code reproduction',
        'ensemble':False,'knn':False})


def verify(output):
    protocol=read(output/'protocol.json')
    for rel,digest in protocol['guard_sha256'].items():
        if sha256(ROOT/rel)!=digest:
            raise ValueError(f'Protected source/data/V4 artifact changed: {rel}')
    return protocol


def archive_partial(path,output):
    """Called only under this experiment's exclusive worker lock."""
    if not path.exists():
        return
    root=output.resolve();source=path.resolve()
    if not source.is_relative_to(root) or source==root:
        raise ValueError('Recovery path is outside experiment.')
    target=root/'interrupted'/path.relative_to(output).parent/(path.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if not target.resolve().is_relative_to(root):
        raise ValueError('Invalid recovery destination.')
    target.parent.mkdir(parents=True,exist_ok=True)
    path.rename(target)


def validation_trial(output,seed,config,settings):
    from .search import run_search
    from ...train import train
    from ...accuracy import calibrate
    from ...predict import evaluate
    selected=run_search(config,settings,seed,output/'activation_search'/f'seed{seed}')
    destination=output/'validation_trials'/f'seed{seed}'
    destination.mkdir(parents=True,exist_ok=True)
    if not (destination/'training/summary.json').exists():
        archive_partial(destination/'training',output)
        train(copy.deepcopy(selected),destination/'training',seed,'cuda')
    if not (destination/'calibration/calibration.json').exists():
        archive_partial(destination/'calibration',output)
        calibrate(destination/'training',destination/'calibration','cuda',hybrid=False)
    checkpoint=destination/'calibration/pure_selected.pt'
    if not all((destination/'validation'/name).exists() for name in ('metrics.json','predictions.csv')):
        archive_partial(destination/'validation',output)
        evaluate(checkpoint,SPLIT,'validation',destination/'validation','cuda')
    summary=read(destination/'training/summary.json');calibration=read(destination/'calibration/calibration.json')
    metrics=read(destination/'validation/metrics.json')
    _,payload=load_checkpoint(checkpoint,torch.device('cpu'))
    if (summary['seed']!=seed or payload['seed']!=seed or metrics['checkpoint_sha256']!=sha256(checkpoint)
        or summary['test_evaluated'] is not False or calibration['test_evaluated'] is not False
        or calibration['source_checkpoint_sha256']!=sha256(destination/'training/best.pt')
        or training_design(read(destination/'training/resolved_config.json'))!=selected):
        raise ValueError('Invalid S-KAN Validation/config provenance.')
    result={'seed':seed,'best_epoch':summary['best_epoch'],'mean_2d':metrics['mean_2d'],
        'decoder':calibration['best_pure'],'checkpoint_sha256':sha256(checkpoint),
        'selected_config_sha256':sha256(output/'activation_search'/f'seed{seed}'/'selected_config.json'),
        'test_evaluated':False}
    immutable(destination/'result.json',result)
    return result


def run(output):
    from ...final_refit import refit
    from ...predict import evaluate
    protocol=verify(output);results=[]
    for seed in SEEDS:
        verify(output)
        save_json(output/'state.json',{'stage':'selection_then_validation','current_seed':seed,'completed':len(results)})
        results.append(validation_trial(output,seed,protocol['config'],protocol['search']))
    sources=[output/'validation_trials'/f'seed{s}'/'calibration/pure_selected.pt' for s in SEEDS]
    lock={'scope':'one_joint_kan_mixer','selection_split':'internal_validation_only','test_evaluated':False,
        'seeds':SEEDS,'final_refit_epochs':refit_epochs(results),'all_ten_validation_results':results,
        'source_sha256':protocol['guard_sha256'],'split_manifest_sha256':protocol['split_manifest_sha256'],
        'allowed_source_checkpoint_sha256':[sha256(p) for p in sources],
        'training_configs_by_source_sha256':{sha256(p):{'path':str(p.parent.parent/'training/resolved_config.json'),
            'sha256':sha256(p.parent.parent/'training/resolved_config.json')} for p in sources}}
    immutable(output/'selection_lock.json',lock)
    for seed,source in zip(SEEDS,sources):
        verify(output)
        save_json(output/'state.json',{'stage':'refit','current_seed':seed,'epochs':lock['final_refit_epochs']})
        destination=output/'final_refit'/f'seed{seed}'
        if not (destination/'summary.json').exists():
            archive_partial(destination,output)
            refit(source,output/'selection_lock.json',destination,seed,'cuda')
    checkpoints=[output/'final_refit'/f'seed{s}'/'best.pt' for s in SEEDS]
    hashes=[sha256(p) for p in checkpoints];states=[state_digest(p) for p in checkpoints]
    if len(set(hashes))!=10 or len(set(states))!=10:
        raise ValueError('Ten independently trained model states are required.')
    immutable(output/'test_checkpoint_lock.json',{'test_evaluated_at_creation':False,
        'selection_lock_sha256':sha256(output/'selection_lock.json'),
        'checkpoints':[{'seed':s,'path':str(p),'sha256':h,'model_state_sha256':d}
                       for s,p,h,d in zip(SEEDS,checkpoints,hashes,states)]})
    per_seed=[]
    for seed,checkpoint,digest in zip(SEEDS,checkpoints,hashes):
        verify(output)
        if sha256(checkpoint)!=digest:
            raise ValueError('Refit checkpoint changed after freezing.')
        save_json(output/'state.json',{'stage':'official_test','current_seed':seed})
        destination=output/'official_test'/f'seed{seed}'
        if not all((destination/name).exists() for name in ('metrics.json','predictions.csv')):
            archive_partial(destination,output)
            evaluate(checkpoint,SPLIT,'test',destination,'cuda')
        per_seed.append(audit_test(checkpoint,destination,SPLIT))
    aggregate={key:{'mean':statistics.mean(r[key] for r in per_seed),
                    'sample_std':statistics.stdev(r[key] for r in per_seed)} for key in METRICS}
    baseline=protocol['baseline']['result']['per_seed']
    deltas=[r['mean_2d']-b['mean_2d'] for r,b in zip(per_seed,baseline)]
    immutable(output/'official_test_summary.json',{'variant':'S-KAN-Mixer IPS adaptation',
        'primary_regime':'full_development_refit','n_models':10,'test_rows_per_model':1111,'seeds':SEEDS,
        'refit_epochs':lock['final_refit_epochs'],'per_seed':per_seed,'aggregate':aggregate,
        'paired_mpe_delta_skan_minus_v4':{'mean':statistics.mean(deltas),'sample_std':statistics.stdev(deltas),
            'per_seed':[{'seed':s,'delta_m':d} for s,d in zip(SEEDS,deltas)]},
        'baseline_v4':protocol['baseline']['result'],'historical_test_exposure':True,'no_reselection_after_test':True})
    completion_audit(output)
    from .report import write_report
    write_report(output)
    save_json(output/'state.json',{'stage':'completed','mean_2d':aggregate['mean_2d'],'audited_rows':11110})
    print('SKAN_COMPLETE '+json.dumps(aggregate),flush=True)


def completion_audit(output):
    protocol=verify(output);summary=read(output/'official_test_summary.json')
    lock=read(output/'test_checkpoint_lock.json')
    if len(lock['checkpoints'])!=10 or summary['seeds']!=SEEDS:
        raise ValueError('Incomplete cohort.')
    for seed,record in zip(SEEDS,lock['checkpoints']):
        selection=read(output/'activation_search'/f'seed{seed}'/'selection.json')
        config=read(output/'final_refit'/f'seed{seed}'/'resolved_config.json')
        metadata=read(output/'final_refit'/f'seed{seed}'/'run_metadata.json')
        if (selection['selected_nodes']!=500 or selection['test_evaluated'] or selection['validation_evaluated']
            or selection['weights_transferred_to_training']
            or config['kan']['selected_activations']!=selection['selected_activations']
            or metadata['actual_counts']!={'train':19937,'validation':0}
            or not metadata['original_preprocessing_retained']
            or metadata['test_evaluated_during_training'] or sha256(Path(record['path']))!=record['sha256']):
            raise ValueError('Selection/refit provenance failed completion audit.')
        audit_test(Path(record['path']),output/'official_test'/f'seed{seed}',SPLIT)
    immutable(output/'completion_audit.json',{'status':'passed','seeds':SEEDS,'test_rows_audited':11110,
        'baseline_files_unchanged':True,'source_and_data_guards_passed':True,
        'matched_outer_configuration_verified':True,'selected_nodes_per_model':500,
        'fresh_training_then_fresh_full_development_refit':True,
        'protocol_sha256':sha256(output/'protocol.json'),'test_summary_sha256':sha256(output/'official_test_summary.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','audit'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();output=args.output_dir.resolve()
    if args.action=='prepare':
        prepare(output);return
    if args.action=='audit':
        completion_audit(output);return
    # Windows releases this OS lock even if a worker is interrupted. Existence
    # of state.json or an old PID is never interpreted as a running worker.
    import msvcrt
    with (output/'worker.lock').open('a+b') as stream:
        stream.seek(0)
        if stream.read(1)==b'':
            stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:
            run(output)
        except Exception as error:
            save_json(output/'failure.json',{'type':type(error).__name__,'message':str(error),
                'utc':datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
