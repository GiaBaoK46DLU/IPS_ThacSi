"""Guard against premature test access, conditional errors and seed selection."""
import copy
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ips.data import ROOT
from ips.algorithms.kan.confirmation import SEEDS

spec = importlib.util.spec_from_file_location('kan_under6_final',ROOT/'scripts/kan_under6_final.py')
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


class LockedFinalTests(unittest.TestCase):
    def lock(self):
        rows = [{'name':'winner','seed':seed,'role':'primary','sha256':str(seed)} for seed in SEEDS]
        rows += [{'name':'budget100','seed':seed,'role':'baseline_control','sha256':'base'+str(seed)}
                 for seed in SEEDS[:3]]
        for row in rows:
            row['path'] = 'unused-selected-checkpoint.pt'
        return {'scope':'one_joint_kan_mixer','selection_split':'internal_validation_only','test_evaluated':False,
                'winner':'winner','seeds':SEEDS.copy(),'final_refit_epochs':80,'sources':rows,
                'allowed_source_checkpoint_sha256':[row['sha256'] for row in rows]}

    def test_layout_requires_all_ten_seeds_controls_and_pretest_selection(self):
        lock = self.lock()
        final.validate_layout(lock)
        missing = copy.deepcopy(lock)
        missing['sources'].pop(0)
        with self.assertRaisesRegex(ValueError,'every prespecified'):
            final.validate_layout(missing)
        for key,value in [('test_evaluated',True),('final_refit_epochs',101)]:
            changed = copy.deepcopy(lock)
            changed[key] = value
            with self.assertRaises(ValueError):
                final.validate_layout(changed)

    def test_evaluate_never_loads_test_if_any_refit_is_missing(self):
        lock = self.lock()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(final,'load_selection',return_value=({},lock,output/'lock.json')), \
                    patch.object(final,'load_bundle') as loader:
                with self.assertRaisesRegex(ValueError,'All refits must complete'):
                    final.evaluate(output)
            loader.assert_not_called()
            self.assertFalse((output/'official_test_evaluation_manifest.json').exists())

    def test_coordinate_audit_rejects_conditional_mean_and_reordered_rows(self):
        class Example:
            name = 'test'
            inputs = SimpleNamespace(row_indices=np.array([0,1,2]))
            xy = np.array([[0.,0.],[3.,4.],[6.,8.]])
            def __len__(self): return 3
        dataset = Example()
        columns = ['row_index','true_x','true_y','pred_x','pred_y','error_2d']
        rows = [[0,0,0,0,0,0],[1,3,4,0,0,5],[2,6,8,0,0,10]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions.csv'
            def write(values):
                with path.open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(values)
            write(rows)
            result = final.audit_predictions(dataset,path,{'n_samples':3,'mean_2d':5.},expected_count=3)
            self.assertEqual(result['recomputed_mean_2d'],5.)
            with self.assertRaisesRegex(ValueError,'ALL test rows'):
                final.audit_predictions(dataset,path,{'n_samples':3,'mean_2d':0.},expected_count=3)
            write(rows[::-1])
            with self.assertRaisesRegex(ValueError,'order/count'):
                final.audit_predictions(dataset,path,{'n_samples':3,'mean_2d':5.},expected_count=3)

    def test_aggregation_reports_mean_over_all_models_not_best_seed(self):
        rows = []
        for seed,error in [(42,4.),(43,7.),(44,7.)]:
            row = {'name':'winner','seed':seed,'role':'primary','regime':'full_development_refit'}
            row.update({key:error for key in ['mean_2d','median_2d','p90_2d','rmse_2d']})
            row.update({key:1. for key in ['building_accuracy','floor_accuracy','joint_accuracy']})
            rows.append(row)
        result = final.aggregate(rows,[42,43,44])
        self.assertEqual(result['groups'][0]['mean_2d_mean'],6.)
        self.assertFalse(result['target_met_on_primary_ten_seed_mean'])
        with self.assertRaises(ValueError):
            final.aggregate(rows[:1],[42,43,44])


if __name__ == '__main__':
    unittest.main()
