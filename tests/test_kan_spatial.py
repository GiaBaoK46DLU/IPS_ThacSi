"""Research-v2 invariants: training geometry, rank features and one-model heads."""
import copy
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ips.data import ROOT, sha256
from ips.algorithms.kan import spatial_research
from ips.algorithms.kan.augmentation import NearbyPointMixing, training_representation
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.reference import observed_signal_rank, represent_signal, fit_reference_points
from ips.algorithms.kan.accuracy import decode_reference
from ips.algorithms.kan.runtime import configure_runtime, read_config
from ips.algorithms.kan.train import combined_loss


class SpatialKANTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(42, 'cpu', 2)

    def config(self):
        config = read_config(ROOT / 'src/ips/algorithms/kan/configs/accuracy_spatial.json')
        config['mixer'].update(channels=8, blocks=1, token_hidden=8, channel_hidden=8, dropout=0)
        config['positioning'].update(head_width=8, rssi_noise_db=0, device_shift_db=0, ap_dropout=0)
        xy = np.column_stack([np.arange(26), np.arange(26) + 2]).astype(float)
        data = SimpleNamespace(split_name='train', classes=torch.arange(13).repeat_interleave(2),
                               xy_original=xy, xy=torch.tensor(xy/100, dtype=torch.float32))
        fit_reference_points(config, data)
        return config

    def features(self):
        mask = torch.randint(0, 2, (3, 520)).float()
        return torch.stack([torch.rand(3, 520)*mask, mask], -1)

    def test_average_tied_ranks_are_order_preserving_and_affine_invariant(self):
        z = torch.tensor([[.2,.5,.5,0.],[.7,0.,0.,0.],[0.,0.,0.,0.]])
        mask = torch.tensor([[1.,1.,1.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.]])
        expected = torch.tensor([[-1.,.5,.5,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        torch.testing.assert_close(observed_signal_rank(z, mask), expected)
        torch.testing.assert_close(observed_signal_rank((2*z+.1)*mask, mask), expected)
        encoded = represent_signal(z, mask, {'input_mode':'relative_rank_mask','mask_scale':1.})
        self.assertEqual(encoded.shape, (3,4,3))
        torch.testing.assert_close(encoded[...,1], expected)
        torch.testing.assert_close(encoded[...,2], mask)
        self.assertTrue(torch.isfinite(encoded).all())

    def test_rank_mode_has_1560_inputs_and_one_predictor(self):
        config = self.config()
        config['positioning']['input_mode'] = 'relative_rank_mask'
        with self.assertRaisesRegex(ValueError, 'Tokenization'):
            build_model(config)
        config['mixer']['features_per_token'] = 60
        output = build_model(config).eval()(self.features())
        self.assertEqual(output['logits'].shape, (3,13))
        self.assertEqual(output['xy'].shape, (3,2))
        self.assertTrue(torch.isfinite(output['xy']).all())

    def test_spatial_mix_interpolates_target_with_same_alpha_and_respects_floor_radius(self):
        class Data:
            split_name = 'train'
            classes = torch.tensor([0,0,0,1])
            xy_original = np.array([[0.,0.],[4.,0.],[50.,0.],[1.,0.]])
            features = torch.tensor([[[.2,1.],[0.,0.]],[[.8,1.],[.5,1.]],
                                     [[.9,1.],[1.,1.]],[[1.,1.],[1.,1.]]])
            def __len__(self): return 4
            def __getitem__(self,index):
                return self.features[index],self.classes[index],torch.tensor(self.xy_original[index],dtype=torch.float32)
        data = Data()
        before = data.features.clone()
        sampler = NearbyPointMixing(data, 1., 5.)
        for _ in range(20):
            features, label, xy = sampler[0]
            self.assertEqual(label.item(), 0)
            self.assertGreaterEqual(xy[0].item(), .8-1e-6)
            self.assertLessEqual(xy[0].item(), 3.2+1e-6)
            self.assertAlmostEqual(features[0,0].item(), .2+.15*xy[0].item(), places=6)
            self.assertIn(features[1,1].item(), (0.,1.))
            self.assertEqual(xy[1].item(), 0.)
        torch.testing.assert_close(data.features,before,rtol=0,atol=0)
        torch.testing.assert_close(sampler[2][0],data.features[2],rtol=0,atol=0)
        for split in ('validation','test','development'):
            data.split_name = split
            with self.assertRaisesRegex(ValueError,'requires train'):
                NearbyPointMixing(data,1.,5.)
        data.split_name = 'development'
        NearbyPointMixing(data,1.,5.,final_refit=True)

    def test_augmented_training_factory_rejects_ambiguous_combinations(self):
        with self.assertRaisesRegex(ValueError,'alternative'):
            training_representation(None,{'nearby_point_mixing':.5,'same_point_mixing':.5})
        with self.assertRaisesRegex(ValueError,'alternative'):
            training_representation(None,{'nearby_point_mixing':.5,'same_point_consistency_weight':1.})
        with self.assertRaisesRegex(ValueError,r'in \[0,1\]'):
            training_representation(None,{'nearby_point_mixing':2.})

    def screening_fixture(self, output):
        (output/'configs').mkdir()
        (output/'configs/candidate.json').write_text('{}',encoding='utf-8')
        (output/'protocol.json').write_text(json.dumps({'candidate_order':['candidate']}),encoding='utf-8')
        plan = {'protocol_sha256':sha256(output/'protocol.json'), 'source_sha256':{'src/example.py':'original'},
                'screening_candidates_config_sha256':{'candidate':sha256(output/'configs/candidate.json')}}
        (output/'confirmation_plan.json').write_text(json.dumps(plan),encoding='utf-8')
        return plan

    def test_screening_rejects_changed_sources_before_starting_a_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.screening_fixture(output)
            with patch.object(spatial_research,'source_hashes',return_value={'src/example.py':'changed'}), \
                    patch.object(spatial_research,'trial') as execute:
                with self.assertRaisesRegex(ValueError,'Frozen implementation changed'):
                    spatial_research.screen(output)
                execute.assert_not_called()

    def test_completed_screening_cannot_be_relabelled_as_a_new_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            plan = self.screening_fixture(output)
            directory = output/'trials/candidate/seed42'
            directory.mkdir(parents=True)
            (directory/'completed.json').write_text('{}',encoding='utf-8')
            (directory/'stage0_source_hashes.json').write_text(json.dumps({'src/example.py':'old'}),encoding='utf-8')
            with patch.object(spatial_research,'source_hashes',return_value=plan['source_sha256']), \
                    patch.object(spatial_research,'completed',return_value={}), \
                    patch.object(spatial_research,'trial') as execute:
                with self.assertRaisesRegex(ValueError,'Implementation recorded'):
                    spatial_research.screen(output)
                execute.assert_not_called()

    def test_residual_starts_at_zero_and_receives_coordinate_gradient(self):
        config = self.config()
        config['positioning'].update(coordinate_residual_scale=.05,coordinate_residual_width=8)
        model = build_model(config)
        output = model(self.features())
        torch.testing.assert_close(output['coordinate_residual'],torch.zeros(3,13,2),rtol=0,atol=0)
        labels = torch.tensor([0,1,2])
        target = torch.tensor([[.2,.3],[.3,.4],[.4,.5]])
        loss,_,_ = combined_loss(output,labels,target,config['loss'])
        loss.backward()
        gradient = model.coordinate_residual[-1].base_weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(),0.)

    def test_residual_is_bounded_and_preserved_by_calibration_and_state_restore(self):
        config = self.config()
        config['positioning'].update(coordinate_residual_scale=.05,coordinate_residual_width=8)
        model = build_model(config).eval()
        with torch.no_grad(): model.coordinate_residual[-1].base_weight.fill_(.1)
        features = self.features()
        output = model(features)
        self.assertTrue((output['coordinate_residual'].abs() <= .05).all())
        decoded = decode_reference(output['reference_logits'],model,5,1.,
                                   coordinate_residual=output['coordinate_residual'])
        torch.testing.assert_close(output['xy'],decoded['xy'],rtol=0,atol=1e-7)
        clone = build_model(copy.deepcopy(config)).eval()
        clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone(features)['xy'],output['xy'],rtol=0,atol=0)
        config['positioning']['coordinate_residual_scale'] = float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            build_model(config)


if __name__ == '__main__':
    unittest.main()
