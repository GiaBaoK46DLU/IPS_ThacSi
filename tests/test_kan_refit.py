"""Post-selection refit must faithfully retain the selected training method."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ips.data import ROOT, META_COLUMNS, sha256
from ips.algorithms.kan.accuracy import calibrated_payload
from ips.algorithms.kan.augmentation import SamePointMixing
from ips.algorithms.kan.final_refit import recover_training_config, refit, validate_refit_bundle
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.reference import fit_reference_points
from ips.algorithms.kan.runtime import configure_runtime, load_checkpoint, read_config
from ips.algorithms.kan.train import update_ema


class KANRefitTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(42,'cpu',2)

    def fixture(self):
        config = read_config(ROOT/'src/ips/algorithms/kan/configs/accuracy_spatial.json')
        config['mixer'].update(channels=8,blocks=1,token_hidden=8,channel_hidden=8,dropout=0)
        config['positioning'].update(head_width=8,rssi_noise_db=0,device_shift_db=0,ap_dropout=0,top_k=2)
        labels = torch.arange(13).repeat_interleave(3)
        points = np.column_stack([labels.numpy(),labels.numpy()+2.]).astype(float)
        data = SimpleNamespace(split_name='train',classes=labels,xy_original=points,
                               xy=torch.tensor(points/100,dtype=torch.float32))
        fit_reference_points(config,data)
        return config,data

    def test_legacy_calibration_requires_hashed_original_and_rejects_changes(self):
        config,_ = self.fixture()
        selected = copy.deepcopy(config)
        selected['positioning']['temperature'] = .5
        source = {'config':selected,'metadata':{'config':selected,'validation_inference_calibration':{}}}
        with self.assertRaisesRegex(ValueError,'hashed original'):
            recover_training_config(source,{},'source-hash')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'original.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            declared = {'path':str(path),'sha256':sha256(path)}
            lock = {'training_configs_by_source_sha256':{'source-hash':declared}}
            restored,evidence = recover_training_config(source,lock,'source-hash')
            self.assertEqual(restored['positioning']['temperature'],1.)
            self.assertEqual(evidence['kind'],'hashed_original_json')
            selected['mixer']['blocks'] = 3
            with self.assertRaisesRegex(ValueError,'beyond inference'):
                recover_training_config(source,lock,'source-hash')
            path.write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash differs'):
                recover_training_config(source,lock,'source-hash')

    def test_training_decoder_uses_original_topk_temperature_but_eval_uses_frozen_values(self):
        config,_ = self.fixture()
        # Two reference points per class, with a known interpolation result.
        config['positioning']['references'] = [
            {'class_id':c,'xy':[float(c)+i,0.]} for c in range(13) for i in (0,1)]
        config['positioning'].update(top_k=1,temperature=.5,training_top_k=2,training_temperature=1.)
        model = build_model(config)
        scores = torch.tensor([[3.,1.]*13])
        training = model._decode(scores)
        expected_x = torch.softmax(torch.tensor([3.,1.]),0)[1]
        torch.testing.assert_close(training['xy'][0,0],expected_x)
        model.eval()
        inference = model._decode(scores)
        torch.testing.assert_close(inference['xy'],torch.zeros(1,2),rtol=0,atol=0)

    def test_augmentation_needs_explicit_development_role_for_final_refit(self):
        _,data = self.fixture()
        data.split_name = 'development'
        with self.assertRaisesRegex(ValueError,'requires train'):
            SamePointMixing(data,.5)
        SamePointMixing(data,.5,final_refit=True)
        data.split_name = 'validation'
        with self.assertRaises(ValueError):
            SamePointMixing(data,.5,final_refit=True)

    def test_refit_requires_the_frozen_split_and_training_coordinate_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'manifest.json'
            manifest.write_text('{}',encoding='utf-8')
            bundle = SimpleNamespace(directory=root,classes=[{'class_id':0,'building':0,'floor':0}],
                preprocessing={'coordinates':{'center':[0.,0.],'shared_scale':100.}},ap_order=['WAP001'])
            source = {'split_manifest_sha256':sha256(manifest),'classes':copy.deepcopy(bundle.classes),
                'preprocessing':copy.deepcopy(bundle.preprocessing),'ap_order':bundle.ap_order.copy()}
            lock = {'scope':'one_joint_kan_mixer','split_manifest_sha256':sha256(manifest)}
            validate_refit_bundle(source,bundle,lock)
            source['preprocessing']['coordinates']['center'][0] = 1.
            with self.assertRaisesRegex(ValueError,'preprocessing'):
                validate_refit_bundle(source,bundle,lock)
            source['preprocessing'] = copy.deepcopy(bundle.preprocessing)
            manifest.write_text('{"modified":true}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'manifest differ'):
                validate_refit_bundle(source,bundle,lock)

    def test_refit_preserves_mixing_ema_training_decoder_and_no_heldout_evaluation(self):
        self.run_paired_refit(0.)

    def test_refit_also_retains_paired_reference_consistency(self):
        self.run_paired_refit(1.)

    def run_paired_refit(self,consistency_weight):
        config,data = self.fixture()
        config['training'].update(same_point_mixing=.5,ema_decay=.995,batch_size=16)
        if consistency_weight:
            config['training']['same_point_consistency_weight'] = consistency_weight
        classes = json.loads((ROOT/'data/processed/uji_group_split_v1/class_mapping.json').read_text())
        metadata = np.zeros((39,9),dtype=np.float64)
        metadata[:,:2] = data.xy_original
        for row,label in enumerate(data.classes.tolist()):
            metadata[row,2] = classes[label]['floor']
            metadata[row,3] = classes[label]['building']
        development = SimpleNamespace(size=39,rssi=np.full((39,520),-52,dtype=np.int16),metadata=metadata,
            pairs=[(classes[c]['building'],classes[c]['floor']) for c in data.classes.tolist()],
            column=lambda name:metadata[:,META_COLUMNS.index(name)])
        bundle = SimpleNamespace(directory=ROOT/'data/processed/uji_group_split_v1',development=development,
            preprocessing={'coordinates':{'center':[0.,0.],'shared_scale':100.}},classes=classes,
            ap_order=[f'WAP{i:03d}' for i in range(1,521)])
        checkpoint = {'format_version':1,'config':config,'metadata':{'config':config},'classes':classes,
                      'model_state_dict':build_model(config).state_dict()}
        selected = calibrated_payload(checkpoint,{'top_k':1,'temperature':.5,'mean_2d':5.4},{})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'selected.pt'
            torch.save(selected,source)
            lock = root/'lock.json'
            lock.write_text(json.dumps({'selection_split':'internal_validation_only','test_evaluated':False,
                'final_refit_epochs':1,'allowed_source_checkpoint_sha256':[sha256(source)]}))
            with patch('ips.algorithms.kan.final_refit.load_bundle',return_value=bundle), \
                    patch('ips.algorithms.kan.augmentation.SamePointMixing',wraps=SamePointMixing) as mixing, \
                    patch('ips.algorithms.kan.final_refit.update_ema',wraps=update_ema) as averaged:
                result = refit(source,lock,root/'refit',42,'cpu')
            mixing.assert_called_once()
            self.assertTrue(mixing.call_args.kwargs['final_refit'])
            self.assertEqual(mixing.call_args.args[0].split_name,'development')
            self.assertEqual(averaged.call_count,3)
            model,payload = load_checkpoint(root/'refit/best.pt',torch.device('cpu'))
            self.assertEqual(result['train_samples'],39)
            self.assertIsNone(payload['validation_mean_2d'])
            self.assertEqual(payload['config']['positioning']['training_temperature'],1.)
            self.assertEqual(payload['config']['positioning']['training_top_k'],2)
            self.assertEqual(payload['config']['positioning']['temperature'],.5)
            self.assertEqual(payload['config']['positioning']['top_k'],1)
            self.assertEqual(payload['ema_updates'],3)
            self.assertEqual(payload['metadata']['selection_weights'],'train_step_ema')
            self.assertEqual(payload['metadata']['training_views_per_dataset_row'],2 if consistency_weight else 1)
            self.assertFalse(payload['metadata']['test_evaluated_during_training'])
            self.assertTrue(any(not torch.equal(payload['model_state_dict'][key],value)
                for key,value in payload['training_model_state_dict'].items()))
            self.assertTrue(all(torch.isfinite(value).all() for value in model.state_dict().values()))


if __name__ == '__main__':
    unittest.main()
