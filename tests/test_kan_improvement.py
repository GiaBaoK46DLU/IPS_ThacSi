"""Scientific invariants for opt-in joint-building KAN variants."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ips.data import ROOT
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.reference import fit_reference_points, represent_signal
from ips.algorithms.kan.runtime import read_config, configure_runtime, load_checkpoint
from ips.algorithms.kan.accuracy import calibrated_payload, decode_reference
from ips.algorithms.kan.improvement import trial, validate_kan_only
from ips.algorithms.kan.train import combined_loss, reference_consistency_loss, update_ema


class JointKANImprovementTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(42, "cpu", 2)

    def config(self):
        c = read_config(ROOT / 'src/ips/algorithms/kan/configs/accuracy_spatial.json')
        c['mixer'].update(channels=8, blocks=1, token_hidden=8, channel_hidden=8, dropout=0)
        c['positioning'].update(head_width=8, rssi_noise_db=0, device_shift_db=0, ap_dropout=0)
        xy = np.column_stack([np.arange(26), np.arange(26)+2]).astype(float)
        train = SimpleNamespace(split_name='train', classes=torch.arange(13).repeat_interleave(2),
                                xy_original=xy, xy=torch.tensor(xy/100, dtype=torch.float32))
        fit_reference_points(c, train)
        return c, train

    def test_default_representation_retains_exact_legacy_values(self):
        z, m = torch.rand(3, 520), torch.randint(0, 2, (3, 520)).float()
        for scale in (0., 1.):
            torch.testing.assert_close(represent_signal(z, m, {'mask_scale': scale}),
                                       torch.stack([z, m*scale], -1), rtol=0, atol=0)

    def test_ema_matches_hand_computed_update_and_preserves_buffers(self):
        model = torch.nn.Linear(2, 1, bias=False)
        shadow = copy.deepcopy(model).requires_grad_(False)
        model.register_buffer('catalogue', torch.tensor([3.]))
        shadow.register_buffer('catalogue', torch.tensor([0.]))
        model.weight.data.fill_(2)
        shadow.weight.data.zero_()
        update_ema(shadow, model, .995, 0)
        torch.testing.assert_close(shadow.weight, torch.full_like(shadow.weight, 1.8))
        torch.testing.assert_close(shadow.catalogue, model.catalogue)
        self.assertIsNone(shadow.weight.grad)

    def test_same_point_mixing_never_changes_targets_or_reads_heldout(self):
        from ips.algorithms.kan.augmentation import SamePointMixing
        class Example:
            split_name = 'train'
            classes = torch.tensor([0,0,1])
            xy_original = np.array([[0.,0.],[0.,0.],[100.,100.]])
            features = torch.tensor([[[.4,1.],[0.,0.]],[[.8,1.],[.5,1.]],[[1.,1.],[1.,1.]]])
            def __len__(self): return 3
            def __getitem__(self,i): return self.features[i],self.classes[i],torch.tensor(self.xy_original[i])
        data = Example()
        original = data.features.clone()
        mixed = SamePointMixing(data, 1.)
        for _ in range(10):
            x,c,xy = mixed[0]
            self.assertEqual(c.item(),0)
            torch.testing.assert_close(xy,torch.zeros(2,dtype=torch.float64))
            self.assertTrue(.4 <= x[0,0] <= .8)
            self.assertIn(x[1,1].item(),(0.,1.))
        torch.testing.assert_close(data.features, original,rtol=0,atol=0)
        torch.testing.assert_close(mixed[2][0],data.features[2],rtol=0,atol=0)
        data.split_name='validation'
        with self.assertRaisesRegex(ValueError,'requires train'):
            SamePointMixing(data,1.)

    def test_cross_phone_pairs_keep_location_targets_and_never_use_other_locations(self):
        from ips.algorithms.kan.augmentation import SamePointPairs
        class Example:
            split_name = 'train'
            classes = torch.tensor([0,0,0,1])
            xy_original = np.array([[0.,0.],[0.,0.],[0.,0.],[100.,100.]])
            metadata = {'PHONEID':np.array([1,1,2,2])}
            features = torch.tensor([[[.1,1.]],[[.3,1.]],[[.7,1.]],[[1.,1.]]])
            def __len__(self): return 4
            def __getitem__(self,i): return self.features[i],self.classes[i],torch.tensor(self.xy_original[i])
        data = Example()
        pairs = SamePointPairs(data,partner_policy='different_phone_first')
        for _ in range(10):
            x,label,xy,partner = pairs[0]
            torch.testing.assert_close(partner,data.features[2],rtol=0,atol=0)
            torch.testing.assert_close(x,data.features[0],rtol=0,atol=0)
            self.assertEqual(label.item(),0)
            torch.testing.assert_close(xy,torch.zeros(2,dtype=torch.float64))
        torch.testing.assert_close(pairs[3][3],data.features[3],rtol=0,atol=0)
        data.split_name = 'validation'
        with self.assertRaises(ValueError):
            SamePointPairs(data)

    def test_reference_consistency_is_symmetric_zero_for_identical_and_differentiable(self):
        same = torch.randn(4,13)
        self.assertAlmostEqual(reference_consistency_loss(torch.cat([same,same]),4).item(),0.,places=6)
        scores = torch.tensor([[2.,-1.],[-1.,2.]],requires_grad=True)
        loss = reference_consistency_loss(scores,1)
        self.assertGreater(loss.item(),0.)
        self.assertLess(loss.item(),np.log(2))
        torch.testing.assert_close(loss,reference_consistency_loss(scores.flip(0),1))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        with self.assertRaises(ValueError):
            reference_consistency_loss(scores,2)

    def test_single_kan_scope_rejects_memory_ensembles_and_multiview(self):
        cfg, _ = self.config()
        validate_kan_only(cfg)
        for change in ({'head_type':'metric_memory'}, {'inference_shifts_db':[-3.,0.,3.]}):
            changed = copy.deepcopy(cfg)
            changed['positioning'].update(change)
            with self.assertRaisesRegex(ValueError, 'KAN-Mixer only'):
                validate_kan_only(changed)
            with self.assertRaises(ValueError):
                build_model(changed)
        for field in ('ensemble_members','fingerprint_refinement'):
            with self.assertRaisesRegex(ValueError, 'excluded'):
                validate_kan_only(cfg, {field:[]})
        model = build_model(cfg)
        self.assertEqual(type(model).__name__, 'ReferenceKANMixer')
        self.assertFalse(hasattr(model, 'memory_embeddings'))

    def test_calibration_cannot_mutate_shared_training_config(self):
        cfg, _ = self.config()
        checkpoint = {'config':cfg, 'metadata':{'config':cfg}}
        candidate = {'top_k':10, 'temperature':.5, 'mean_2d':5.4}
        calibration = {'selection_split':'internal_validation_only', 'selected':candidate}
        payload = calibrated_payload(checkpoint,candidate,calibration)
        self.assertIs(checkpoint['config'], checkpoint['metadata']['config'])
        self.assertEqual(checkpoint['config']['positioning']['temperature'],1.)
        self.assertEqual(payload['config']['positioning']['temperature'],.5)
        self.assertEqual(payload['metadata']['config']['positioning']['temperature'],1.)
        self.assertEqual(payload['metadata']['training_config']['positioning']['temperature'],1.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'calibrated.pt'
            torch.save(payload,path)
            restored = torch.load(path,map_location='cpu',weights_only=False)
            self.assertIsNot(restored['config'], restored['metadata']['config'])

    def test_calibration_cli_does_not_enable_hybrid_by_default(self):
        from ips.algorithms.kan import accuracy
        for extra in ([], ['--pure-only']):
            with patch('sys.argv', ['accuracy','--run-dir','train','--output-dir','out',*extra]), \
                    patch.object(accuracy,'calibrate') as calibrated:
                accuracy.main()
            self.assertFalse(calibrated.call_args.kwargs['hybrid'])

    def test_research_rejects_memory_before_creating_a_run(self):
        cfg, _ = self.config()
        cfg['positioning']['head_type'] = 'metric_memory'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output/'configs').mkdir()
            (output/'configs/rejected.json').write_text(json.dumps(cfg),encoding='utf-8')
            (output/'protocol.json').write_text('{"split_manifest_sha256":"unused"}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'KAN-Mixer only'):
                trial(output,'rejected')
            self.assertFalse((output/'trials').exists())

    def test_relative_features_cancel_common_device_offset_and_handle_absence(self):
        mask = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
        signal = torch.tensor([[.4, .7, 0.], [0., 0., 0.]])
        setting = {'input_mode': 'absolute_relative_mask', 'mask_scale': 1.}
        encoded = represent_signal(signal, mask, setting)
        shifted = represent_signal(signal+.1*mask, mask, setting)
        torch.testing.assert_close(encoded[..., 1], shifted[..., 1])
        self.assertTrue(torch.isfinite(encoded).all())
        torch.testing.assert_close(encoded[0, 2], torch.zeros(3))
        torch.testing.assert_close(encoded[1], torch.zeros(3, 3))
        torch.testing.assert_close(encoded[..., 0], signal)

    def test_input_modes_pooling_auxiliary_and_losses_have_finite_gradients(self):
        for mode, pooling, auxiliary in [('absolute_relative_mask', 'flatten', False),
                                         ('relative_mask', 'learned', True)]:
            config, train = self.config()
            config['positioning'].update(input_mode=mode, pooling=pooling, mask_scale=1.,
                                         auxiliary_classifier=auxiliary, top_k=2)
            config['mixer']['features_per_token'] = 60 if mode == 'absolute_relative_mask' else 40
            config['loss']['coordinate_source'] = 'topk'
            model = build_model(config)
            x = torch.rand(4, 520, 2)
            x[..., 1] = (x[..., 1] > .5).float()
            out = model(x)
            loss, _, _ = combined_loss(out, train.classes[:4], train.xy[:4], config['loss'])
            loss.backward()
            self.assertEqual(out['logits'].shape, (4, 13))
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            model.eval()
            with torch.inference_mode():
                direct = model(x)
                decoded = decode_reference(direct['reference_logits'], model, 2, 1.,
                                            direct.get('auxiliary_logits'))
            torch.testing.assert_close(direct['xy'], decoded['xy'], rtol=0, atol=0)
            torch.testing.assert_close(direct['logits'], decoded['logits'], rtol=0, atol=0)

    def test_invalid_token_count_or_mode_rejected_and_checkpoint_is_self_contained(self):
        cfg, _ = self.config()
        cfg['positioning']['input_mode'] = 'absolute_relative_mask'
        with self.assertRaisesRegex(ValueError, '3 features'):
            build_model(cfg)
        cfg['mixer']['features_per_token'] = 60
        cfg['positioning']['auxiliary_classifier'] = True
        model = build_model(cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'new.pt'
            torch.save({'format_version':1, 'config':cfg, 'classes':[{}]*13,
                        'model_state_dict':model.state_dict()}, checkpoint)
            restored, _ = load_checkpoint(checkpoint, torch.device('cpu'))
            x = torch.rand(2,520,2)
            with torch.inference_mode():
                torch.testing.assert_close(model(x)['xy'], restored(x)['xy'], rtol=0, atol=0)

    def test_kan_prediction_head_is_internal_and_has_finite_gradients(self):
        from ips.algorithms.kan.models.vendor.efficient_kan import KANLinear
        cfg, data = self.config()
        cfg['positioning']['head_transform'] = 'kan'
        model = build_model(cfg)
        self.assertIsInstance(model.head[0], KANLinear)
        self.assertIsInstance(model.head[2], KANLinear)
        self.assertEqual(sum(isinstance(layer,KANLinear) for layer in model.modules()),6)
        x = torch.rand(4,520,2)
        x[...,1] = (x[...,1]>.5).float()
        output = model(x)
        loss,_,_ = combined_loss(output,data.classes[:4],data.xy[:4],cfg['loss'])
        loss.backward()
        for name,parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad,name)
            self.assertTrue(torch.isfinite(parameter.grad).all(),name)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'kan_head.pt'
            torch.save({'format_version':1,'config':cfg,'classes':[{}]*13,
                        'model_state_dict':model.state_dict()},path)
            restored,_ = load_checkpoint(path,torch.device('cpu'))
            with torch.inference_mode():
                torch.testing.assert_close(model(x)['xy'],restored(x)['xy'],rtol=0,atol=0)

    def test_validation_support_uses_original_labels_order_and_actual_metre_errors(self):
        from ips.algorithms.kan.improvement_analysis import coordinate_support
        train = SimpleNamespace(name='train',class_ids=np.array([0,1]),xy=np.array([[0.,0.],[8.,10.]]))
        validation = SimpleNamespace(name='validation',class_ids=np.array([0,1]),xy=np.array([[0.,0.],[10.,10.]]),
            inputs=SimpleNamespace(row_indices=np.array([5,7])),
            info=SimpleNamespace(classes=[{'building':0,'floor':0},{'building':1,'floor':0}]))
        columns = ['row_index','true_x','true_y','pred_x','pred_y','error_2d',
                   'true_building','true_floor','pred_building','pred_floor']
        rows = [[5,0,0,3,4,5,0,0,0,0],[7,10,10,10,10,0,1,0,1,0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions.csv'
            def write(values):
                with path.open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(values)
            write(rows)
            result = coordinate_support(train,validation,path)
            self.assertEqual(result['overall']['mean_2d'],2.5)
            self.assertEqual(result['seen_train_coordinate']['mean_2d'],5.)
            self.assertEqual(result['unseen_train_coordinate']['mean_2d'],0.)
            write(rows[::-1])
            with self.assertRaisesRegex(ValueError,'order'):
                coordinate_support(train,validation,path)
            rows[0][5] = 4
            write(rows)
            with self.assertRaisesRegex(ValueError,'Saved errors'):
                coordinate_support(train,validation,path)

    def test_confirmation_ranks_all_fixed_seeds_and_preserves_original_plan(self):
        from ips.algorithms.kan.confirmation import immutable_json, rank_confirmed
        result = rank_confirmed({'best_single_seed':{42:4.,43:8.,44:8.},
                                 'stable':{42:5.,43:5.,44:5.}},[42,43,44])
        self.assertEqual(result[0]['name'],'stable')
        self.assertEqual(result[0]['n'],3)
        self.assertEqual(result[0]['sample_std_validation_2d'],0.)
        for scores in ({'missing':{42:4.}}, {'invalid':{42:float('nan'),43:5.,44:5.}}):
            with self.assertRaises(ValueError):
                rank_confirmed(scores,[42,43,44])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'
            immutable_json(path,{'seeds':[42,43,44]})
            original = path.read_bytes()
            immutable_json(path,{'seeds':[42,43,44]})
            with self.assertRaises(FileExistsError):
                immutable_json(path,{'seeds':[42]})
            self.assertEqual(path.read_bytes(),original)

    def test_balanced_confirmation_preserves_full_error_but_protects_unseen_coordinates(self):
        from ips.algorithms.kan.confirmation import rank_balanced
        seeds = [42,43,44]
        scores = {'better_on_seen':{seed:{'full':5.4,'seen':4.9,'unseen':7.2} for seed in seeds},
                  'better_on_unseen':{seed:{'full':5.6,'seen':5.3,'unseen':6.6} for seed in seeds}}
        ranked = rank_balanced(scores,seeds)
        self.assertEqual(ranked[0]['name'],'better_on_unseen')
        self.assertEqual(ranked[0]['mean_full_validation_2d'],5.6)
        self.assertAlmostEqual(ranked[0]['mean_balanced_selection_score'],5.95)
        with self.assertRaises(ValueError):
            rank_balanced({'partial':{42:scores['better_on_unseen'][42]}},seeds)


if __name__ == '__main__':
    unittest.main()
