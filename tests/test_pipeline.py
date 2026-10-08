"""Kiểm tra các lỗi có thể làm sai thí nghiệm: trục mixing, inverse, checkpoint."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ips.data import ROOT, feature_transform, sha256
from ips.metrics import evaluate_predictions, inverse_xy
from ips.models import build_model, parameter_count
from ips.models.networks import MixerBlock
from ips.models.vendor.efficient_kan import KANLinear
from ips.predict import read_fingerprints
from ips.runtime import configure_runtime, load_checkpoint, read_config
from ips.train import combined_loss


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_runtime(42, "cpu", 2)
        cls.config = read_config(ROOT / "configs/kan_mixer.json")

    def test_missing_and_weak_signal_remain_distinct(self):
        result = feature_transform(np.asarray([[-52, 100, -104, 0]], dtype=np.int16))
        np.testing.assert_array_equal(result, [[[0.5, 1], [0, 0], [0, 1], [1, 1]]])

    def test_kan_does_not_require_personal_baselines_or_mlp_config(self):
        self.assertNotIn("mlp", self.config)
        with tempfile.TemporaryDirectory() as tmp, patch("ips.data.ROOT", Path(tmp)):
            self.assertEqual(parameter_count(build_model(self.config)), 365199)
            for name in ("mlp", "mlp_mixer"):
                with self.assertRaisesRegex(ValueError, "Personal MLP baselines are excluded"):
                    build_model({"model": name})

    def test_token_and_channel_axes_with_hand_computed_permutations(self):
        x = torch.tensor([[[1., 10.], [2., 20.], [3., 30.]]])
        block = MixerBlock(3, 2, 3, 2, 0, self.config["kan"])
        block.token_norm = block.channel_norm = nn.Identity()
        token = nn.Linear(3, 3, bias=False)
        token.weight.data.copy_(torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]))
        channel = nn.Linear(2, 2, bias=False)
        channel.weight.data.zero_()
        block.token_mixer, block.channel_mixer = token, channel
        expected = torch.tensor([[[4.,40.],[3.,30.],[5.,50.]]])
        torch.testing.assert_close(block(x), expected)
        token.weight.data.zero_()
        channel.weight.data.copy_(torch.tensor([[0.,1.],[1.,0.]]))
        torch.testing.assert_close(block(x), torch.tensor([[[11.,11.],[22.,22.],[33.,33.]]]))

    def test_kan_model_loss_gradients_and_design_parameter_count(self):
        expected_counts = {"kan_mixer":365199}
        x = torch.rand(2, 520, 2)
        for name, expected in expected_counts.items():
            with self.subTest(model=name):
                config = copy.deepcopy(self.config); config["model"] = name
                model = build_model(config)
                self.assertEqual(parameter_count(model), expected)
                result = model(x)
                self.assertEqual(result["logits"].shape, (2,13))
                self.assertEqual(result["xy"].shape, (2,2))
                loss, _, _ = combined_loss(result, torch.tensor([0,12]), torch.ones(2,2), config["loss"])
                loss.backward()
                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())

    def test_kan_spline_basis_and_base_outside_support(self):
        layer = KANLinear(2, 3, grid_range=[-3,3])
        x = torch.tensor([[0., 1.], [-2., 2.]])
        torch.testing.assert_close(layer.b_splines(x).sum(dim=-1), torch.ones_like(x))
        layer.spline_weight.data.zero_()
        layer.base_weight.data.fill_(1)
        outside = torch.tensor([[[20., 30.]]])
        expected = nn.functional.silu(outside).sum(dim=-1,keepdim=True).expand(1,1,3)
        torch.testing.assert_close(layer(outside), expected)
        torch.testing.assert_close(layer(outside), layer(outside.reshape(-1,2)).reshape(1,1,3))

    def test_inverse_preserves_metre_error_on_large_coordinates(self):
        preprocessing = {"coordinates":{"center":[-7500.,4864800.],"shared_scale":100.}}
        actual = inverse_xy([[.03,.04]], preprocessing)
        pairs = [(b,f) for b in range(3) for f in range(5 if b==2 else 4)]
        classes = [{"class_id":i,"building":b,"floor":f} for i,(b,f) in enumerate(pairs)]
        metrics = evaluate_predictions(np.array([0]),np.array([0]),np.array([[-7500.,4864800.]]), actual, classes)
        self.assertAlmostEqual(metrics["mean_2d"], 5.0)
        self.assertAlmostEqual(metrics["rmse_2d"], 5.0)
        self.assertEqual(metrics["mae_xy"], [3.,4.])

    def test_checkpoint_round_trip_kan_backbone(self):
        x = torch.rand(2,520,2)
        with tempfile.TemporaryDirectory(prefix="ips_checkpoint_") as tmp:
            for name in ["kan_mixer"]:
                config=copy.deepcopy(self.config); config["model"]=name
                original=build_model(config).eval()
                payload={"format_version":1,"config":config,"classes":[{}]*13,
                         "model_state_dict":original.state_dict()}
                path=Path(tmp)/(name+'.pt'); torch.save(payload,path)
                loaded,_=load_checkpoint(path,torch.device('cpu'))
                with torch.inference_mode():
                    first,second=original(x),loaded(x)
                torch.testing.assert_close(first['logits'],second['logits'],rtol=0,atol=0)
                torch.testing.assert_close(first['xy'],second['xy'],rtol=0,atol=0)

    def test_inference_needs_only_rssi_and_rejects_reordered_aps(self):
        from ips.data import AP_COLUMNS, csv_bytes
        with tempfile.TemporaryDirectory(prefix="ips_input_") as tmp:
            path=Path(tmp)/'input.csv'
            path.write_bytes(csv_bytes(AP_COLUMNS, [[100]*519+[-52]]))
            self.assertEqual(read_fingerprints(path,AP_COLUMNS).shape,(1,520))
            swapped=AP_COLUMNS.copy(); swapped[0],swapped[1]=swapped[1],swapped[0]
            path.write_bytes(csv_bytes(swapped,[[100]*520]))
            with self.assertRaises(ValueError): read_fingerprints(path,AP_COLUMNS)

    def test_vendored_source_is_exact_pinned_commit(self):
        directory=ROOT/'src/ips/models/vendor'
        provenance=json.loads((directory/'provenance.json').read_text(encoding='utf8'))
        self.assertEqual(len(provenance['commit']),40)
        for name,record in provenance['files'].items():
            self.assertEqual(sha256(directory/name),record['sha256'])


if __name__ == "__main__":
    unittest.main()
