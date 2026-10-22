"""Prespecified multi-seed VALIDATION confirmation for one joint KAN-Mixer.

This tool never performs official-test inference. It freezes a refit/evaluation
plan only after all validation runs complete. Every prediction model remains
an individual KAN-Mixer; seed repetitions are not an ensemble.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from ...data import ROOT, sha256
from .improvement import trial, validate_kan_only
from .runtime import save_json


SEEDS = [42,43,44,100,2024,777,999,45,46,47]


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise FileExistsError(f'Frozen research artifact differs: {path}')
    else:
        save_json(path,value)


def source_hashes():
    files = list((ROOT/'src/ips/algorithms/kan').rglob('*.py'))
    files += [ROOT/'src/ips'/name for name in ['data.py','metrics.py','cli.py','__main__.py']]
    files += list((ROOT/'src/ips/data_access').rglob('*.py'))
    return {path.relative_to(ROOT).as_posix():sha256(path) for path in sorted(files)}


def prepare(output):
    configs = {}
    for path in sorted((output/'configs').glob('*.json')):
        if path.name.endswith('.design.json'):
            continue
        config = json.loads(path.read_text(encoding='utf-8'))
        try:
            validate_kan_only(config)
        except ValueError:
            continue
        if not 1 <= config['training']['max_epochs'] <= 100:
            raise ValueError('Every candidate must keep the agreed maximum of 100 epochs.')
        configs[path.stem] = sha256(path)
    if 'budget100' not in configs:
        raise ValueError('Confirmation requires the baseline training-budget control.')
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    immutable_json(output/'confirmation_plan.json',{
        'plan_version':3,'scope':'one_joint_kan_mixer','selection_split':'internal_validation_only',
        'test_evaluated':False,'seeds':SEEDS,'initial_confirmation_seeds':[42,43,44],
        'screening_candidates_config_sha256':configs,'source_sha256':source_hashes(),
        'protocol_sha256':sha256(output/'protocol.json'),
        'split_manifest_sha256':protocol['split_manifest_sha256'],
        'first_stage':'top two seed42 full-validation means, plus best unseen-coordinate validation mean, plus budget100',
        'winner_rule':'lowest mean over seeds42,43,44 of (seen-coordinate mean2D + unseen-coordinate mean2D)/2; full mean then name break ties',
        'selection_score_is_not_reported_full_validation_or_test_metric':True,
        'checkpoint_and_decoder_selection':'unchanged full-validation mean2D; only architecture confirmation uses balanced support',
        'rationale':'Validation has3053 seen-coordinate and859 unseen-coordinate rows; protect generalization before any new test access.',
        'remaining_seeds':'confirm winner on the remaining seven prespecified seeds; no reselection by test',
        'refit_epoch_rule':'rounded median validation-best epoch over all10 winner seeds, bounded1..100',
        'test_policy':{'primary':'mean of per-model mean2D over ten independently trained joint KAN-Mixers',
                       'n_test_rows_per_model':1111,'sample_std_ddof':1,'target_mean_2d_m':6.,
                       'canonical_deployment_seed_fixed_before_test':42,
                       'report_train_only_and_final_refit_separately':True,
                       'no_ensembles_no_seed_selection_by_test':True,
                       'historical_test_exposure':True}})


def rank_confirmed(scores, seeds):
    """Use every prespecified seed; never pick a favorable subset."""
    rows = []
    for name,values in scores.items():
        if set(values) != set(seeds):
            raise ValueError(f'Incomplete prespecified confirmation seeds for {name}.')
        observations = [float(values[seed]) for seed in seeds]
        if not all(math.isfinite(value) and value >= 0 for value in observations):
            raise ValueError('Validation scores must be finite nonnegative measured errors.')
        rows.append({'name':name,'n':len(seeds),'mean_validation_2d':statistics.mean(observations),
                     'sample_std_validation_2d':statistics.stdev(observations) if len(seeds)>1 else None,
                     'per_seed':{str(seed):values[seed] for seed in seeds}})
    return sorted(rows,key=lambda row:(row['mean_validation_2d'],row['name']))


def support_scores(output,name,seed):
    directory = output/'trials'/name/f'seed{seed}'
    support = json.loads((directory/'validation_support.json').read_text(encoding='utf-8'))
    if (support['checkpoint_sha256'] != sha256(directory/'calibration/pure_selected.pt')
            or support['overall']['n'] != 3912
            or support['seen_train_coordinate']['n'] != 3053
            or support['unseen_train_coordinate']['n'] != 859):
        raise ValueError('Validation support analysis changed checkpoint, rows or coordinate groups.')
    return {'full':support['overall']['mean_2d'],
            'seen':support['seen_train_coordinate']['mean_2d'],
            'unseen':support['unseen_train_coordinate']['mean_2d'],
            'balanced':(support['seen_train_coordinate']['mean_2d']+
                        support['unseen_train_coordinate']['mean_2d'])/2}


def rank_balanced(scores,seeds):
    rows = []
    for name,values in scores.items():
        if set(values) != set(seeds):
            raise ValueError('Balanced confirmation requires every fixed seed.')
        for item in values.values():
            if not all(math.isfinite(item[key]) and item[key]>=0 for key in ('full','seen','unseen')):
                raise ValueError('Coordinate-support errors must be finite measured validation values.')
        balanced = [(values[seed]['seen']+values[seed]['unseen'])/2 for seed in seeds]
        rows.append({'name':name,'n':len(seeds),'mean_balanced_selection_score':statistics.mean(balanced),
            'sample_std_balanced_selection_score':statistics.stdev(balanced) if len(seeds)>1 else None,
            'mean_full_validation_2d':statistics.mean(values[seed]['full'] for seed in seeds),
            'mean_seen_validation_2d':statistics.mean(values[seed]['seen'] for seed in seeds),
            'mean_unseen_validation_2d':statistics.mean(values[seed]['unseen'] for seed in seeds),
            'per_seed':{str(seed):values[seed] for seed in seeds}})
    return sorted(rows,key=lambda row:(row['mean_balanced_selection_score'],row['mean_full_validation_2d'],row['name']))


def completed(output,name,seed):
    destination = output/'trials'/name/f'seed{seed}'
    path = destination/'completed.json'
    if not path.exists():
        if destination.exists():
            raise RuntimeError(f'Incomplete trial exists; inspect it before resuming: {destination}')
        trial(output,name,seed)
    result = json.loads(path.read_text(encoding='utf-8'))
    if (result['seed'] != seed or result['name'] != name or result.get('test_evaluated') is not False
            or result['best_validation']['kind'] != 'pure_reference_kan_mixer'
            or not math.isfinite(result['best_validation']['mean_2d'])
            or result['best_validation']['mean_2d'] < 0):
        raise ValueError(f'Invalid confirmation trial evidence: {path}')
    launch = json.loads((destination/'launch.json').read_text(encoding='utf-8'))
    if launch['config_sha256'] != sha256(output/'configs'/f'{name}.json'):
        raise ValueError('Trial configuration hash changed after launch.')
    return result


def confirm(output):
    plan_path = output/'confirmation_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    def check_frozen_sources():
        if source_hashes() != plan['source_sha256']:
            raise ValueError('KAN/data code changed after the confirmation plan; review before continuing.')
        if sha256(output/'protocol.json') != plan['protocol_sha256']:
            raise ValueError('Original research protocol changed.')
        for name,digest in plan['screening_candidates_config_sha256'].items():
            if sha256(output/'configs'/f'{name}.json') != digest:
                raise ValueError(f'Frozen screening configuration changed: {name}')
    check_frozen_sources()
    from .improvement_analysis import analyze_completed
    ranking = []
    for name in plan['screening_candidates_config_sha256']:
        path = output/'trials'/name/'seed42/completed.json'
        if not path.exists():
            raise ValueError(f'Finish all prespecified screening trials before confirmation: {name}')
        result = completed(output,name,42)
        ranking.append({'name':name,'validation_mean_2d':result['best_validation']['mean_2d'],
                        'evidence_sha256':sha256(path)})
    ranking.sort(key=lambda row:(row['validation_mean_2d'],row['name']))
    analyze_completed(output)
    support_ranking = [{'name':row['name'],**support_scores(output,row['name'],42)} for row in ranking]
    best_unseen = min(support_ranking,key=lambda row:(row['unseen'],row['full'],row['name']))['name']
    candidates = sorted({row['name'] for row in ranking[:2]}|{'budget100',best_unseen})
    immutable_json(output/'confirmation_candidates.json',{
        'selection_split':'internal_validation_only','test_evaluated':False,
        'plan_sha256':sha256(plan_path),'screening_ranking':ranking,
        'coordinate_support_ranking':support_ranking,'best_unseen_candidate':best_unseen,
        'candidate_names':candidates})
    for name in candidates:
        for seed in plan['initial_confirmation_seeds']:
            check_frozen_sources()
            completed(output,name,seed)
    analyze_completed(output)
    scores = {name:{seed:support_scores(output,name,seed)
                    for seed in plan['initial_confirmation_seeds']} for name in candidates}
    confirmed = rank_balanced(scores,plan['initial_confirmation_seeds'])
    winner = confirmed[0]['name']
    immutable_json(output/'confirmed_architecture.json',{
        'selection_split':'internal_validation_only','test_evaluated':False,
        'plan_sha256':sha256(plan_path),'ranking':confirmed,'winner':winner})
    for seed in plan['seeds']:
        check_frozen_sources()
        completed(output,winner,seed)
    final_scores = {winner:{seed:completed(output,winner,seed)['best_validation']['mean_2d']
                           for seed in plan['seeds']}}
    summary = {'selection_split':'internal_validation_only','test_evaluated':False,
               'winner':winner,'results':rank_confirmed(final_scores,plan['seeds']),
               'architecture_selection_score':'equal-weight seen/unseen-coordinate validation groups; not the test metric',
               'confirmation_plan_sha256':sha256(plan_path)}
    immutable_json(output/'confirmation_summary.json',summary)
    epochs = []
    for seed in plan['seeds']:
        path = output/'trials'/winner/f'seed{seed}/training/summary.json'
        epochs.append(json.loads(path.read_text(encoding='utf-8'))['best_epoch'])
    refit_epochs = max(1,min(100,math.floor(statistics.median(epochs)+.5)))
    sources,training_configs = [],{}
    # Same-epoch full-development controls separate model/training changes from
    # adding validation rows after hyperparameter selection.
    arms = [(winner,seed,'primary') for seed in plan['seeds']]
    if winner != 'budget100':
        arms += [('budget100',seed,'baseline_control') for seed in plan['initial_confirmation_seeds']]
    for name,seed,role in arms:
        directory = output/'trials'/name/f'seed{seed}'
        path = directory/'calibration/pure_selected.pt'
        config_path = directory/'training/resolved_config.json'
        digest = sha256(path)
        training_configs[digest] = {'path':str(config_path),'sha256':sha256(config_path)}
        sources.append({'name':name,'seed':seed,'role':role,'path':str(path),'sha256':digest})
    immutable_json(output/'selection_lock.json',{
        'scope':'one_joint_kan_mixer','selection_split':'internal_validation_only','test_evaluated':False,
        'confirmation_summary_sha256':sha256(output/'confirmation_summary.json'),
        'source_sha256':plan['source_sha256'],'winner':winner,'seeds':plan['seeds'],
        'split_manifest_sha256':plan['split_manifest_sha256'],
        'final_refit_epochs':refit_epochs,'refit_epoch_rule':plan['refit_epoch_rule'],
        'allowed_source_checkpoint_sha256':[row['sha256'] for row in sources],
        'training_configs_by_source_sha256':training_configs,'sources':sources,
        'full_development_training_rows':19937,'original_preprocessing_retained':True,
        'test_policy':plan['test_policy']})
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Frozen refit epochs={refit_epochs}; official test has NOT been evaluated.',flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','confirm'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    (prepare if args.stage=='prepare' else confirm)(output)


if __name__ == '__main__':
    main()
