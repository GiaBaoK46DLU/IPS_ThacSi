"""Stage-7 correctness: metric definitions, seed SD, alignment and unfair suites."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ips.analysis import audit_suite, check_predictions, mean_std
from ips.benchmark import latency_statistics
from ips.data import ROOT, sha256
from ips.metrics import evaluate_predictions


class EvaluationTests(unittest.TestCase):
    classes = [{'class_id': 0, 'building': 0, 'floor': 0},
               {'class_id': 1, 'building': 1, 'floor': 1}]

    def test_coordinate_mae_differs_from_euclidean_mean_and_threshold_is_inclusive(self):
        result = evaluate_predictions([0, 1], [0, 0], [[0, 0], [0, 0]], [[3, 4], [0, 8]], self.classes)
        self.assertEqual(result['mean_2d'], 6.5)
        self.assertEqual(result['mae_coordinates'], 3.75)
        self.assertAlmostEqual(result['rmse_2d'], np.sqrt(44.5))
        self.assertAlmostEqual(result['rmse_coordinates'], np.sqrt(22.25))
        self.assertEqual(result['thresholds']['8']['fraction_2d_within'], 1)
        self.assertEqual(result['thresholds']['8']['fraction_joint_correct_and_2d_within'], .5)
        self.assertEqual(result['conditional_mean_2d_joint_correct'], 5)

    def test_no_correct_class_has_no_conditional_error(self):
        result = evaluate_predictions([0], [1], [[0, 0]], [[3, 4]], self.classes)
        self.assertIsNone(result['conditional_mean_2d_joint_correct'])
        self.assertEqual(result['conditional_count_joint_correct'], 0)
        for predictions in [([], [], [], []), ([0], [0], [[0, 0]], [[float('nan'), 0]])]:
            with self.assertRaises(ValueError):
                evaluate_predictions(*predictions, self.classes)

    def test_sample_sd_and_single_seed_are_not_confidence_intervals(self):
        self.assertEqual(mean_std([1, 2, 3]), {'n': 3, 'mean': 2, 'std': 1})
        self.assertIsNone(mean_std([1])['std'])
        with self.assertRaises(ValueError):
            mean_std([1, float('nan')])

    def test_latency_throughput_and_amortization(self):
        stats = latency_statistics([.1, .2, .3], 64)
        self.assertAlmostEqual(stats['mean_ms'], 200)
        self.assertAlmostEqual(stats['throughput_samples_per_second'], 320)
        self.assertAlmostEqual(stats['amortized_mean_ms_per_sample'], 3.125)

    def test_prediction_alignment_rejects_swapped_rows(self):
        dataset = SimpleNamespace(indices=torch.tensor([10, 20]), xy_original=np.asarray([[0., 0.], [3., 4.]]),
                                  classes=torch.tensor([0, 1]), metadata={'USERID': np.asarray([1, 2]), 'PHONEID': np.asarray([3, 4])})
        rows = [dict(row_index=str(index), csv_line_number=str(index+2), true_x=str(x), true_y=str(y),
                     pred_x=str(x), pred_y=str(y), true_building=str(label), true_floor=str(label),
                     pred_building=str(label), pred_floor=str(label), class_id=str(label), error_2d='0',
                     USERID=str(user), PHONEID=str(phone))
                for index, x, y, label, user, phone in [(10, 0, 0, 0, 1, 3), (20, 3, 4, 1, 2, 4)]]
        self.assertEqual(check_predictions(rows, dataset, self.classes)['mean_2d'], 0)
        with self.assertRaises(ValueError):
            check_predictions(rows[::-1], dataset, self.classes)

    def test_suite_audit_rejects_wrong_seeds_budget_and_changed_core(self):
        protocol = {'models': ['mlp', 'mlp_mixer', 'kan_mixer'], 'seeds': [42, 43, 44], 'device': 'cuda',
                    'config': {'training': {'max_epochs': 100}},
                    'training_implementation_sha256': {'src/ips/metrics.py': sha256(ROOT/'src/ips/metrics.py')}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'frozen_config.json').write_text(json.dumps(protocol['config']))
            for key, value in [('seeds', [42]), ('device', 'cpu'), ('config', {'training': {'max_epochs': 3}}),
                               ('training_implementation_sha256', {'src/ips/metrics.py': 'wrong'})]:
                changed = copy.deepcopy(protocol)
                changed[key] = value
                (root/'protocol_lock.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    audit_suite(root)

    def test_completed_suite_rejects_smoke_and_unfair_metadata(self):
        # Synthetic temporary audit fixtures, never experimental results/checkpoints.
        config = {'model': 'mlp', 'training': {'max_epochs': 100, 'batch_size': 64, 'min_delta': .01}}
        hashes = {'src/ips/metrics.py': sha256(ROOT/'src/ips/metrics.py')}
        protocol = {'models': ['mlp', 'mlp_mixer', 'kan_mixer'], 'seeds': [42, 43, 44], 'device': 'cuda',
                    'config': config, 'training_implementation_sha256': hashes, 'split_manifest_sha256': 'fixture-split'}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            def save(path, value):
                path.write_text(json.dumps(value), encoding='utf-8')
            save(root/'protocol_lock.json', protocol)
            save(root/'frozen_config.json', config)
            for model in protocol['models']:
                for seed in protocol['seeds']:
                    directory = root/f'{model}_seed{seed}'
                    directory.mkdir()
                    current = copy.deepcopy(config)
                    current['model'] = model
                    metadata = {'run_type': 'training', 'seed': seed, 'config': current,
                                'resolved_max_epochs': 100, 'actual_counts': {'train': 16025, 'validation': 3912},
                                'split_manifest_sha256': 'fixture-split', 'training_implementation_sha256': hashes,
                                'test_evaluated_during_training': False, 'software': {'device': 'cuda', 'threads': 4}}
                    summary = {'run_type': 'training', 'seed': seed, 'model': model, 'status': 'completed',
                               'epochs_completed': 1, 'optimizer_steps': 251, 'best_epoch': 1,
                               'best_validation_mean_2d': 1., 'training_seconds': 1., 'test_evaluated': False}
                    save(directory/'run_metadata.json', metadata)
                    save(directory/'resolved_config.json', current)
                    save(directory/'summary.json', summary)
                    save(directory/'resources.json', {'software': metadata['software'], 'training_phase_seconds': 1.})
                    save(directory/'best_validation_metrics.json', {'mean_2d': 1.})
                    (directory/'history.csv').write_text('epoch,val_mean_2d,epoch_seconds\n1,1,1\n')
                    (directory/'best.pt').write_bytes(b'unit fixture, not a model checkpoint')
            self.assertEqual(audit_suite(root)['status'], 'passed')
            path = root/'mlp_seed42/run_metadata.json'
            original = json.loads(path.read_text())
            for key, value in [('run_type', 'smoke_check'), ('actual_counts', {'train': 256, 'validation': 128}),
                               ('resolved_max_epochs', 3), ('software', {'device': 'cpu', 'threads': 4})]:
                changed = copy.deepcopy(original)
                changed[key] = value
                save(path, changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    audit_suite(root)
                save(path, original)


if __name__ == '__main__':
    unittest.main()
