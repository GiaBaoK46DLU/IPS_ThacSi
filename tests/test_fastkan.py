import copy
import json
import math
import unittest

import torch

from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.fastkan import FastKANLayer, RadialBasisFunction, fastkan_options
from ips.data import ROOT


def v4_config(**kan):
    config = json.loads((ROOT / "src/ips/algorithms/kan/configs/v4_balanced_points.json").read_text(encoding="utf-8"))
    config["kan"].update(kan)
    config["positioning"]["references"] = [{"class_id": c % 13, "xy": [0.0, 0.0]} for c in range(26)]
    config["positioning"]["fit_scope"] = "internal_train_only"
    return config


class FastKANTests(unittest.TestCase):
    def test_gaussian_basis_matches_formula(self):
        rbf = RadialBasisFunction(-2.0, 2.0, 5)
        x = torch.tensor([0.3])
        h = 1.0
        expected = torch.tensor([math.exp(-((0.3 - c) / h) ** 2) for c in (-2, -1, 0, 1, 2)])
        self.assertTrue(torch.allclose(rbf(x)[0], expected, atol=1e-6))

    def test_layer_equals_spline_plus_base_path(self):
        torch.manual_seed(0)
        layer = FastKANLayer(3, 4, {"fastkan": {"num_grids": 6}})
        x = torch.randn(7, 3)
        basis = layer.rbf(layer.layernorm(x)).reshape(7, -1)
        manual = basis @ layer.spline_linear.weight.T + layer.base_linear(torch.nn.functional.silu(x))
        self.assertTrue(torch.allclose(layer(x), manual, atol=1e-6))
        self.assertEqual(layer(torch.randn(2, 5, 3)).shape, (2, 5, 4))

    def test_options_reject_unknown_and_invalid(self):
        with self.assertRaises(ValueError):
            fastkan_options({"fastkan": {"grid_count": 8}})
        with self.assertRaises(ValueError):
            fastkan_options({"fastkan": {"num_grids": 1}})
        self.assertEqual(fastkan_options({})["num_grids"], 8)

    def test_mixer_uses_fastkan_only_when_requested(self):
        x = torch.rand(2, 520, 2)
        x[..., 1] = (x[..., 0] > 0.5).float()
        base = build_model(v4_config(), 13)
        fast = build_model(v4_config(backend="fastkan"), 13)
        self.assertFalse(any(isinstance(m, FastKANLayer) for m in base.modules()))
        self.assertEqual(sum(isinstance(m, FastKANLayer) for m in fast.modules()), 8)
        self.assertEqual(fast(x)["xy"].shape, (2, 2))



class DecoderAndInputTests(unittest.TestCase):
    def test_building_floor_mixture_is_posterior_mean_in_best_building(self):
        from ips.algorithms.kan.models.reference import building_floor_mixture
        buildings = torch.tensor([0, 0, 1])
        logits = torch.log(torch.tensor([[0.3, 0.3, 0.4]]))
        conditional = torch.tensor([[[0., 0.], [2., 2.], [9., 9.]]])
        # Building 0 has 0.6 > 0.4, so it is chosen even though class 2 is the argmax class.
        self.assertTrue(torch.allclose(building_floor_mixture(logits, conditional, buildings), torch.tensor([[1., 1.]])))

    def test_default_decoder_unchanged_and_mixture_requires_buildings(self):
        config = v4_config()
        self.assertEqual(build_model(config, 13).position_decoder, "argmax_class")
        config["positioning"]["position_decoder"] = "building_floor_mixture"
        with self.assertRaises(ValueError):
            build_model(config, 13)
        config["positioning"]["class_buildings"] = [0] * 4 + [1] * 4 + [2] * 5
        model = build_model(config, 13).eval()
        x = torch.rand(3, 520, 2)
        x[..., 1] = (x[..., 0] > 0.5).float()
        out = model(x)
        from ips.algorithms.kan.models.reference import building_floor_mixture
        expected = building_floor_mixture(
            out["logits"], out["conditional_xy"], model.class_buildings)
        self.assertTrue(torch.allclose(out["xy"], expected))

    def test_rank_mask_is_invariant_to_monotone_distortion(self):
        from ips.algorithms.kan.models.reference import represent_signal
        signal = torch.tensor([[0.2, 0.0, 0.5, 0.4]])
        mask = torch.tensor([[1., 0., 1., 1.]])
        distorted = torch.sqrt(signal) * mask
        a = represent_signal(signal, mask, {"input_mode": "rank_mask"})
        b = represent_signal(distorted, mask, {"input_mode": "rank_mask"})
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
