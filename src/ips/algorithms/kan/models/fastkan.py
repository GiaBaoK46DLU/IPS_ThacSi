"""FastKAN layer (Gaussian radial basis KAN) for the IPS KAN-Mixer.

Independent re-implementation of the FastKAN layer described by Ziyao Li,
"Kolmogorov-Arnold Networks are Radial Basis Function Networks" (2024,
arXiv:2405.06721; reference code github.com/ZiyaoLi/fast-kan, Apache-2.0).

    phi_j(u)  = exp(-((u - c_j) / h)^2),  c_j = linspace(grid_min, grid_max, G),
                                          h   = (grid_max - grid_min) / (G - 1)
    y         = W_s · vec(phi(LayerNorm(x)))  +  W_b · SiLU(x)

The B-spline basis of efficient-kan is replaced by G Gaussian bumps on a fixed
grid. LayerNorm (over the layer's input features) keeps the inputs inside the
grid, which is why FastKAN does not need grid updates. Defaults follow the
reference code: G=8 on [-2, 2], spline weights ~ trunc-normal(std=0.1), and a
SiLU base path implemented as a biased nn.Linear.
"""
from __future__ import annotations

import math

import torch
from torch import nn

DEFAULTS = {"num_grids": 8, "grid_min": -2.0, "grid_max": 2.0, "use_layernorm": True,
            "spline_weight_init_scale": 0.1, "use_base_update": True, "trainable_grid": False}


def fastkan_options(kan: dict) -> dict:
    options = {**DEFAULTS, **kan.get("fastkan", {})}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown FastKAN options: {sorted(unknown)}")
    if type(options["num_grids"]) is not int or options["num_grids"] < 2:
        raise ValueError("FastKAN num_grids must be an integer >= 2.")
    if not options["grid_min"] < options["grid_max"]:
        raise ValueError("FastKAN grid_min must be below grid_max.")
    if options["spline_weight_init_scale"] <= 0:
        raise ValueError("FastKAN spline_weight_init_scale must be positive.")
    return options


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float, grid_max: float, num_grids: int, trainable: bool = False):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=trainable)
        self.denominator = (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator).square())


class FastKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, kan: dict):
        super().__init__()
        options = fastkan_options(kan)
        self.in_features, self.out_features = in_features, out_features
        self.layernorm = nn.LayerNorm(in_features) if options["use_layernorm"] else None
        self.rbf = RadialBasisFunction(options["grid_min"], options["grid_max"], options["num_grids"],
                                       options["trainable_grid"])
        self.spline_linear = nn.Linear(in_features * options["num_grids"], out_features, bias=False)
        nn.init.trunc_normal_(self.spline_linear.weight, mean=0.0, std=options["spline_weight_init_scale"])
        self.base_linear = nn.Linear(in_features, out_features) if options["use_base_update"] else None

    def forward(self, x):
        normalized = self.layernorm(x) if self.layernorm is not None else x
        basis = self.rbf(normalized)
        result = self.spline_linear(basis.reshape(*basis.shape[:-2], -1))
        if self.base_linear is not None:
            result = result + self.base_linear(nn.functional.silu(x))
        return result
