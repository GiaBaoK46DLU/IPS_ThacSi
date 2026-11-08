"""No hidden V4 changes, train-only selection, and complete ten-seed locks."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from ips.algorithms.kan.variants.selectable.experiment import (
    SEEDS,CONFIG,V4_CONFIG,read,refit_epochs,validate_matched)
from ips.algorithms.kan.variants.selectable.search import stratified_indices,subset_dataset


class SelectableProtocolTests(unittest.TestCase):
    def test_outer_v4_configuration_must_match_exactly(self):
        baseline=read(V4_CONFIG);config=read(CONFIG)
        validate_matched(config,baseline)
        for section,key,value in [('training','learning_rate',.002),('mixer','channels',128),
                                   ('loss','lambda_xy',1),('positioning','input_mode','absolute_mask')]:
            changed=copy.deepcopy(config);changed[section][key]=value
            with self.assertRaises(ValueError):
                validate_matched(changed,baseline)

    def test_every_seed_is_required_before_refit_epoch_selection(self):
        rows=[{'seed':s,'best_epoch':e} for s,e in zip(SEEDS,[20,30,40,50,60,20,40,30,60,50])]
        self.assertEqual(refit_epochs(rows),40)
        for invalid in (rows[:3],rows[:-1],list(reversed(rows)),rows+[rows[0]]):
            with self.assertRaises(ValueError):
                refit_epochs(invalid)

    def test_train_selection_is_stratified_deterministic_and_not_heldout(self):
        labels=np.repeat(np.arange(13),[1000]*12+[4025])
        a=stratified_indices(labels,.01,42);b=stratified_indices(labels,.01,42)
        self.assertEqual(len(a),160)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(np.unique(labels[a])),13)
        self.assertEqual(len(np.unique(a)),160)
        self.assertFalse(np.array_equal(a,stratified_indices(labels,.01,43)))
        for role in ('validation','test','development'):
            with self.assertRaises(ValueError):
                subset_dataset(SimpleNamespace(split_name=role),a)


if __name__=='__main__':
    unittest.main()
