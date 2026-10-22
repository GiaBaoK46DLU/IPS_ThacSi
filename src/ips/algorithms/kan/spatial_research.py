"""Second validation-only study after closing the first under-6m attempt.

All changes are opt-in parts of ONE KAN-Mixer or train-only augmentation.
This module cannot evaluate the official test. The existing confirmation and
locked final driver remain responsible for subsequent selection/evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ...data import ROOT, sha256
from .confirmation import prepare as prepare_confirmation, source_hashes, completed
from .improvement import trial, validate_kan_only, record_scope
from .runtime import save_json


def prepare(output: Path):
    previous = ROOT / 'runs/kan_joint_under6_20261020_v1'
    previous_summary = previous / 'official_test_summary.json'
    closed = json.loads(previous_summary.read_text(encoding='utf-8'))
    if closed.get('test_evaluated') is not True:
        raise ValueError('Close and audit the previous study before starting another one.')
    audit_path = previous / 'independent_full_test_audit_20261021.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['official_test_summary_sha256'] != sha256(previous_summary) or audit['verified_models'] != 26:
        raise ValueError('Previous-study audit differs from the actual final evidence.')
    if output.exists():
        raise FileExistsError(output)
    base_path = previous / 'configs/same_point_relative.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    variants = {
        'v1_relative': {},
        'rank_relative': {'mixer': {'features_per_token':60},
                          'positioning': {'input_mode':'relative_rank_mask'}},
        'nearby_5m': {'training': {'same_point_mixing':0.,'nearby_point_mixing':.5,'nearby_point_radius_m':5.}},
        'nearby_10m': {'training': {'same_point_mixing':0.,'nearby_point_mixing':.5,'nearby_point_radius_m':10.}},
        'residual_axis5m': {'positioning': {'coordinate_residual_scale':.05,'coordinate_residual_width':32}},
        'nearby_residual': {'training': {'same_point_mixing':0.,'nearby_point_mixing':.5,'nearby_point_radius_m':10.},
                            'positioning': {'coordinate_residual_scale':.05,'coordinate_residual_width':32}},
        'relative_dropout': {'positioning': {'rssi_noise_db':4.,'ap_dropout':.15}},
    }
    configs = {}
    for name, changes in variants.items():
        config = copy.deepcopy(base)
        for section, values in changes.items():
            config[section].update(values)
        validate_kan_only(config)
        configs[name] = config
    baseline = json.loads((ROOT / 'src/ips/algorithms/kan/configs/accuracy_spatial.json').read_text(encoding='utf-8'))
    baseline['training']['patience'] = 100
    configs['budget100'] = baseline
    output.mkdir(parents=True)
    (output / 'configs').mkdir()
    for name, config in configs.items():
        save_json(output / 'configs' / f'{name}.json',config)
    save_json(output / 'protocol.json',{
        'study':'spatial_research_v2','scope':'one_joint_kan_mixer',
        'objective':'official test all-row mean2D below6m; joint three buildings /13classes',
        'split_manifest_sha256':sha256(ROOT / base['split_dir'] / 'manifest.json'),
        'previous_closed_test_summary_sha256':sha256(previous_summary),
        'previous_independent_audit_sha256':sha256(audit_path),
        'base_training_configuration_sha256':sha256(base_path),
        'previous_primary_test_mean_2d':audit['recomputed_primary_mean_2d'],
        'historical_test_exposure':True,
        'exploratory_follow_up_after_previous_test':True,
        'new_candidate_test_evaluated_at_plan_creation':False,
        'new_candidate_selection_uses_validation_only':True,
        'selection_split':'internal_validation_only','screening_seed':42,
        'candidate_order':list(configs),'candidate_overrides_from_previous_relative':variants,
        'control_note':'budget100 is the unchanged historical baseline; v1_relative replays the previous selected training configuration.',
        'hypotheses':{
            'rank_relative':'Within-query tied ranks may reduce sensitivity to monotonic RSSI changes; the full input is not claimed invariant.',
            'nearby':'Interpolation of nearby same-floor training signals and targets may improve continuity; synthetic fingerprints are not physically verified measurements.',
            'residual':'A bounded internal two-layer B-spline KAN head may correct point-decoder bias; .05 normalized units bounds EACH coordinate, not radial distance; never uses a separate predictor or query labels.',
            'relative_dropout':'Larger train-only AP dropout/noise may improve robustness; no guarantee of a test gain.'},
        'coordinate_metric':'unchanged native-coordinate Euclidean2D; all1111test rows',
        'known_building_or_floor_at_inference':False,
        'no_ensemble_no_retrieval_no_per_building_models':True,
        'test_policy':'same locked ten-seed primary full-development refit protocol; no new test evaluation until selection and all refits complete'})
    record_scope(output)
    prepare_confirmation(output)
    print(f'Prepared {len(configs)} immutable validation candidates in {output}',flush=True)


def screen(output: Path):
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    plan = json.loads((output / 'confirmation_plan.json').read_text(encoding='utf-8'))
    if sha256(output / 'protocol.json') != plan['protocol_sha256']:
        raise ValueError('The validation protocol changed after freezing.')
    for name in protocol['candidate_order']:
        if source_hashes() != plan['source_sha256']:
            raise ValueError('Frozen implementation changed; preserve and review this study before a new plan.')
        if sha256(output / 'configs' / f'{name}.json') != plan['screening_candidates_config_sha256'][name]:
            raise ValueError(f'Frozen candidate changed: {name}')
        directory = output / 'trials' / name / 'seed42'
        if directory.exists():
            if not (directory / 'completed.json').is_file():
                raise ValueError(f'Inspect incomplete run before restarting: {directory}')
            completed(output,name,42)
            for stage in (0,1):
                provenance = json.loads((directory / f'stage{stage}_source_hashes.json').read_text(encoding='utf-8'))
                if any(provenance.get(path) != digest for path,digest in plan['source_sha256'].items()):
                    raise ValueError(f'Implementation recorded for the completed trial differs: {directory}')
            print(f'Already completed {name}/seed42; retaining its original evidence.',flush=True)
            continue
        trial(output,name,42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','screen'])
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else screen)(args.output_dir.resolve())


if __name__ == '__main__':
    main()
