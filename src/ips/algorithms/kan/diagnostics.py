"""Thống kê input từng KANLinear: miền nominal và toàn support của knots."""
import math

import torch

from .models.vendor.efficient_kan import KANLinear


class KANDiagnostics:
    def __init__(self, model):
        self.records = {}
        self.enabled = False
        self.handles = []
        for name, layer in model.named_modules():
            if isinstance(layer, KANLinear):
                self.handles.append(layer.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def record(layer, inputs):
            if not self.enabled:
                return
            with torch.no_grad():
                x = inputs[0].detach().reshape(-1, layer.in_features)
                grid = layer.grid
                k = layer.spline_order
                nominal_low, nominal_high = grid[:, k], grid[:, -k-1]
                support_low, support_high = grid[:, 0], grid[:, -1]
                stats = self.records.setdefault(name, {"n": 0, "sum": 0.0, "sum_sq": 0.0,
                    "min": float("inf"), "max": float("-inf"), "outside_nominal": 0, "outside_support": 0})
                stats["n"] += x.numel()
                # float64 accumulation avoids needless precision loss in summary statistics.
                stats["sum"] += x.double().sum().item()
                stats["sum_sq"] += x.double().square().sum().item()
                stats["min"] = min(stats["min"], x.min().item())
                stats["max"] = max(stats["max"], x.max().item())
                stats["outside_nominal"] += ((x < nominal_low) | (x > nominal_high)).sum().item()
                stats["outside_support"] += ((x < support_low) | (x >= support_high)).sum().item()
        return record

    def reset(self):
        self.records = {}

    def summary(self):
        result = {}
        for name, row in self.records.items():
            n = row["n"]
            mean = row["sum"] / n
            result[name] = {"n_values": n, "mean": mean,
                "std": math.sqrt(max(0, row["sum_sq"] / n - mean * mean)),
                "min": row["min"], "max": row["max"],
                "outside_nominal_fraction": row["outside_nominal"] / n,
                "outside_knot_support_fraction": row["outside_support"] / n}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
