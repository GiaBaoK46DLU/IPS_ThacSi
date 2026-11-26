import json
import math
import unittest

import torch

from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models.grkan import GroupRational, GRKANLayer, SECOND_MOMENT, grkan_options
from ips.data import ROOT


def v4_config(**kan):
    config = json.loads((ROOT / "src/ips/algorithms/kan/configs/v4_balanced_points.json").read_text(encoding="utf-8"))
    config["kan"].update(kan)
    config["positioning"]["references"] = [{"class_id": c % 13, "xy": [0.0, 0.0]} for c in range(26)]
    config["positioning"]["fit_scope"] = "internal_train_only"
    return config


class GRKANTests(unittest.TestCase):
    def test_rational_matches_formula(self):
        rational = GroupRational(3, 3, "identity")
        with torch.no_grad():
            rational.numerator.copy_(torch.tensor([[1., 2, 0, 0, 0, 0]] * 3))
            rational.denominator.copy_(torch.tensor([[0., -1, 0, 0]] * 3))
        x = torch.tensor([[0.5, -2.0, 3.0]])
        expected = (1 + 2 * x) / (1 + (x ** 2).abs())
        self.assertTrue(torch.allclose(rational(x), expected))

    def test_identity_and_silu_initialisation(self):
        x = torch.linspace(-4, 4, 801)[:, None].repeat(1, 4)
        self.assertTrue(torch.equal(GroupRational(4, 2, "identity")(x), x))
        error = (GroupRational(4, 2, "silu")(x) - torch.nn.functional.silu(x)).abs().max()
        self.assertLess(error.item(), 1e-4)

    def test_second_moment_constant(self):
        z = torch.randn(400000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        self.assertAlmostEqual((torch.nn.functional.silu(z) ** 2).mean().item(), SECOND_MOMENT["silu"], places=2)

    def test_groups_share_coefficients(self):
        rational = GroupRational(26, 8, "identity")
        self.assertEqual(rational.numerator.shape, (8, 6))
        self.assertEqual(rational.group_index.tolist(), [i * 8 // 26 for i in range(26)])
        self.assertEqual(rational.group_index.unique().numel(), 8)

    def test_layer_is_linear_of_rational_and_preserves_variance(self):
        torch.manual_seed(0)
        layer = GRKANLayer(128, 64, {}, "silu")
        x = torch.randn(4096, 128)
        self.assertTrue(torch.allclose(layer(x), layer.linear(layer.rational(x)), atol=1e-6))
        self.assertAlmostEqual(layer(x).var().item(), 1.0, delta=0.15)

    def test_options_validation(self):
        with self.assertRaises(ValueError):
            grkan_options({"grkan": {"groups": 8}})
        with self.assertRaises(ValueError):
            grkan_options({"grkan": {"num_groups": 0}})

    def test_mixer_uses_grkan_only_when_requested(self):
        x = torch.rand(2, 520, 2)
        x[..., 1] = (x[..., 0] > 0.5).float()
        base = build_model(v4_config(), 13)
        model = build_model(v4_config(backend="grkan"), 13)
        self.assertFalse(any(isinstance(m, GRKANLayer) for m in base.modules()))
        layers = [m for m in model.modules() if isinstance(m, GRKANLayer)]
        self.assertEqual(len(layers), 8)
        self.assertEqual([l.rational.init for l in layers[:2]], ["identity", "silu"])
        self.assertEqual(model(x)["xy"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
