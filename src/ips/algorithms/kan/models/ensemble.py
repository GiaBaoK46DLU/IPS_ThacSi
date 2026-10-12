"""Frozen reference KAN-Mixer ensemble; no fitting or query labels."""
import torch
from torch import nn


class EnsembleReferenceKANMixer(nn.Module):
    def __init__(self, members, confidence_weighted=True):
        super().__init__()
        if len(members) < 2:
            raise ValueError("Ensemble needs at least two independently trained KAN-Mixers.")
        self.members = nn.ModuleList(members)
        self.confidence_weighted = confidence_weighted
        self.classes = members[0].classes
        if any(m.classes != self.classes for m in members):
            raise ValueError("Ensemble class mappings differ.")

    def forward(self, features):
        outputs = [member(features) for member in self.members]
        probabilities = torch.stack([o["logits"].softmax(1) for o in outputs])
        conditional_xy = torch.stack([o["conditional_xy"] for o in outputs])
        if self.confidence_weighted:
            weights = probabilities / probabilities.sum(0, keepdim=True).clamp_min(1e-12)
            conditional = (weights[..., None] * conditional_xy).sum(0)
        else:
            conditional = conditional_xy.mean(0)
        logits = probabilities.mean(0).clamp_min(1e-12).log()
        predicted = logits.argmax(1)
        return {"logits": logits, "conditional_xy": conditional,
                "xy": conditional[torch.arange(len(features), device=features.device), predicted]}
