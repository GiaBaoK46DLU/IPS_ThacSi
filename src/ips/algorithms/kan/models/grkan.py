"""GR-KAN layer (Group-Rational KAN) for the IPS KAN-Mixer.

Independent re-implementation of the GR-KAN idea from Xingyi Yang and Xinchao
Wang, "Kolmogorov-Arnold Transformer" (ICLR 2025, arXiv:2409.10594). Details
such as the coefficient fit below are this project's choices.

    F_g(x)  = P_g(x) / Q_g(x),   P_g(x) = sum_{i=0..5} a_{g,i} x^i,
                                 Q_g(x) = 1 + | sum_{j=1..4} b_{g,j} x^j |
    y       = W · F(x) + bias,   channel i uses the rational of group g(i)

Every edge function phi_{ji}(x) = W_{ji} F_{g(i)}(x) is a learnable rational
function; the C input channels are split into G contiguous groups that share
the rational coefficients (parameter sharing), followed by one linear map.
Q >= 1, so the "safe" denominator has no poles.

Initialisation:
* first layer of a mixing branch: F = identity (a1 = 1, other coefficients 0);
* second layer: F ~= SiLU, coefficients fitted once by least squares on
  x in [-4, 4] (max abs error 1.2e-5, see tests/test_grkan.py);
* W ~ N(0, 1 / (C_in · E[F(x)^2])) with x ~ N(0, 1) (variance preserving),
  bias = 0.
"""
from __future__ import annotations

import math

import torch
from torch import nn

NUMERATOR_DEGREE, DENOMINATOR_DEGREE = 5, 4
# SiLU fit: P/Q with Q = 1 + |b1 x + ... + b4 x^4|, least squares on [-4, 4].
SILU_NUMERATOR = (3.468418083643301e-06, 0.5000000175058231, 0.249988574898683,
                  0.05301366719130746, 0.005687570955355549, 0.0002603145695701611)
SILU_DENOMINATOR = (-1.4078445336969336e-05, -0.1060203445015971,
                    -1.321591779757404e-06, -0.0005205354712575598)
# E[F(x)^2] for x ~ N(0, 1); used by the variance-preserving initialisation.
SECOND_MOMENT = {"identity": 1.0, "silu": 0.3557755198}
DEFAULTS = {"num_groups": 8, "variance_preserving_init": True}


def grkan_options(kan: dict) -> dict:
    options = {**DEFAULTS, **kan.get("grkan", {})}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown GR-KAN options: {sorted(unknown)}")
    if type(options["num_groups"]) is not int or options["num_groups"] < 1:
        raise ValueError("GR-KAN num_groups must be a positive integer.")
    return options


class GroupRational(nn.Module):
    """Safe rational function P/Q shared by contiguous channel groups."""

    def __init__(self, features: int, groups: int, init: str):
        super().__init__()
        if init not in SECOND_MOMENT:
            raise ValueError(f"Unknown rational initialisation: {init}")
        groups = min(groups, features)
        numerator = torch.zeros(groups, NUMERATOR_DEGREE + 1)
        denominator = torch.zeros(groups, DENOMINATOR_DEGREE)
        if init == "identity":
            numerator[:, 1] = 1.0
        else:
            numerator[:] = torch.tensor(SILU_NUMERATOR)
            denominator[:] = torch.tensor(SILU_DENOMINATOR)
        self.numerator = nn.Parameter(numerator)
        self.denominator = nn.Parameter(denominator)
        # Channel i belongs to group floor(i * G / C); groups may differ by one channel.
        self.register_buffer("group_index", torch.arange(features) * groups // features, persistent=False)
        self.features, self.groups, self.init = features, groups, init

    def forward(self, x):
        a = self.numerator[self.group_index]          # [C, 6]
        b = self.denominator[self.group_index]        # [C, 4]
        powers = [torch.ones_like(x), x]
        for _ in range(NUMERATOR_DEGREE - 1):
            powers.append(powers[-1] * x)
        p = sum(a[:, i] * powers[i] for i in range(NUMERATOR_DEGREE + 1))
        q = 1 + torch.abs(sum(b[:, j - 1] * powers[j] for j in range(1, DENOMINATOR_DEGREE + 1)))
        return p / q


class GRKANLayer(nn.Module):
    """y = Linear(F(x)), F a group-rational function on the input channels."""

    def __init__(self, in_features: int, out_features: int, kan: dict, init: str = "identity"):
        super().__init__()
        options = grkan_options(kan)
        self.rational = GroupRational(in_features, options["num_groups"], init)
        self.linear = nn.Linear(in_features, out_features)
        if options["variance_preserving_init"]:
            std = 1.0 / math.sqrt(in_features * SECOND_MOMENT[init])
            nn.init.normal_(self.linear.weight, 0.0, std)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.rational(x))
