"""V4 scientific invariants: held-out isolation, AP identity and spatial gradients."""
import copy
import io
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from ips.algorithms.kan.v4_methods import fit_ap_groups, point_sampler, spatial_risk, validate_permutation
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.runtime import read_config, configure_runtime
from ips.data import ROOT


class V4Tests(unittest.TestCase):
    def data(self):
        features = torch.zeros(6, 520, 2)
        features[:3, [2, 8], 1] = 1
        features[3:, [4, 7], 1] = 1
        return SimpleNamespace(split_name='train', features=features,
            classes=torch.zeros(6, dtype=torch.long),
            xy_original=np.array([[0, 0]] * 4 + [[1, 1]] * 2),
            __len__=lambda: 6)

    def test_grouping_rejects_heldout_and_retains_frozen_refit_order(self):
        data = self.data()
        config = {'positioning': {'ap_grouping': 'coobservation'}}
        data.split_name = 'validation'
        with self.assertRaises(ValueError):
            fit_ap_groups(config, data)
        data.split_name = 'train'
        fit_ap_groups(config, data)
        order = config['positioning']['ap_permutation']
        validate_permutation(order)
        self.assertEqual(set(order[:4]), {2, 4, 7, 8})
        repeated = {'positioning': {'ap_grouping': 'coobservation'}}
        fit_ap_groups(repeated, data)
        self.assertEqual(order, repeated['positioning']['ap_permutation'])
        frozen = copy.deepcopy(config)
        data.split_name = 'development'
        data.features[:] = 0  # the changed development statistics cannot reorder APs
        fit_ap_groups(config, data, final_refit=True)
        self.assertEqual(config, frozen)

    def test_permutation_rejects_missing_duplicate_or_noninteger_identity(self):
        for order in [list(range(519)), [0] * 520, [float(i) for i in range(520)]]:
            with self.assertRaises(ValueError):
                validate_permutation(order)

    def test_point_sampler_equalizes_points_and_is_reproducible(self):
        class Data:
            split_name = 'train'
            classes = torch.zeros(6, dtype=torch.long)
            xy_original = np.array([[0, 0]] * 4 + [[1, 1]] * 2)
            def __len__(self): return 6
        data = Data()
        settings = {'sampling': 'reference_points'}
        a = point_sampler(data, settings, torch.Generator().manual_seed(42))
        b = point_sampler(data, settings, torch.Generator().manual_seed(42))
        self.assertEqual(a.weights[:4].sum(), a.weights[4:].sum())
        self.assertEqual(list(a), list(b))
        data.split_name = 'test'
        with self.assertRaises(ValueError):
            point_sampler(data, settings, torch.Generator())

    def test_spatial_risk_penalizes_distant_wrong_class_mass_with_gradient(self):
        scores = torch.zeros(1, 3, requires_grad=True)
        output = {'reference_logits': scores, 'reference_xy': torch.tensor([[0., 0.], [1., 0.], [10., 0.]])}
        loss = spatial_risk(output, torch.zeros(1, 2))
        self.assertAlmostEqual(loss.item(), 11 / 3, places=5)
        loss.backward()
        self.assertGreater(scores.grad[0, 2].item(), scores.grad[0, 1].item())
        self.assertLess(scores.grad[0, 0].item(), 0)

    def test_grouped_checkpoint_roundtrip_and_one_ap_shapes(self):
        configure_runtime(42, 'cpu', 2)
        config = read_config(ROOT/'src/ips/algorithms/kan/configs/accuracy_spatial.json')
        config['positioning'].update(references=[{'class_id':c, 'xy':[c/10., 0.]} for c in range(13)],
            fit_scope='internal_train_only', head_width=8)
        config['mixer'].update(channels=8, blocks=1, token_hidden=8, channel_hidden=8)
        features = torch.rand(2, 520, 2)
        features[:, :, 1] = (features[:, :, 1] > .5).float()
        for one_ap in (False, True):
            current = copy.deepcopy(config)
            if one_ap:
                current['mixer'].update(tokens=520, features_per_token=2)
            else:
                current['positioning'].update(ap_grouping='coobservation',
                    ap_permutation=list(reversed(range(520))), ap_permutation_fit_scope='internal_train_only')
            model = build_model(current, 13).eval()
            before = model(features)
            self.assertEqual(before['logits'].shape, (2, 13))
            self.assertEqual(before['xy'].shape, (2, 2))
            buffer = io.BytesIO()
            torch.save({'config':current,'state':model.state_dict()}, buffer)
            buffer.seek(0)
            payload = torch.load(buffer,weights_only=True)
            restored = build_model(payload['config'],13).eval()
            restored.load_state_dict(payload['state'],strict=True)
            torch.testing.assert_close(before['xy'],restored(features)['xy'],rtol=0,atol=0)

    def test_finalization_rejects_seven_seed_or_wrong_winner_before_refit(self):
        from ips.algorithms.kan.research_v4 import finalize, SEEDS
        for rows in [
            [{'seed':s,'name':'candidate','test_evaluated':False} for s in SEEDS[:7]],
            [{'seed':s,'name':'other','test_evaluated':False} for s in SEEDS],
        ]:
            with patch('ips.algorithms.kan.research_v4.verify', return_value={}):
                with self.assertRaisesRegex(ValueError, 'all ten validation-only'):
                    finalize(Path('not_used'), 'candidate', rows)


if __name__ == '__main__':
    unittest.main()
