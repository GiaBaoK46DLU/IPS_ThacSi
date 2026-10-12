"""Explicit hybrid: KAN-Mixer + a train-only weighted fingerprint catalogue.

The catalogue is part of the checkpoint. It is not fitted on evaluation inputs.
The neural contribution is reported explicitly; this is not a pure neural net.
"""
import torch
from torch import nn


def db_features(features):
    """(RSSI+105)/105, zero for missing; Euclidean-equivalent to -105 dBm fill."""
    return (features[..., 0] * (104 / 105) + features[..., 1] / 105)


def refine_predictions(output, query_features, catalogue, settings):
    signal = db_features(query_features)
    train_signal, train_xy, train_class = catalogue
    squared = (signal.square().sum(1, keepdim=True) + train_signal.square().sum(1)[None]
               - 2 * signal @ train_signal.T).clamp_min(0)
    k = min(int(settings["neighbors"]), len(train_signal))
    distances, indices = squared.topk(k, largest=False)
    weights = (distances.sqrt() + 1e-4).pow(-float(settings["distance_power"]))
    neighbor_labels = train_class[indices]
    votes = torch.stack([(weights * (neighbor_labels == c)).sum(1)
                         for c in range(output["logits"].shape[1])], 1)
    neighbor_probability = votes / votes.sum(1, keepdim=True)
    strength = float(settings["neighbor_class_weight"])
    probabilities = ((1 - strength) * output["logits"].softmax(1)
                     + strength * neighbor_probability)
    predicted = probabilities.argmax(1)
    floor_distances = squared.masked_fill(train_class[None] != predicted[:, None], float("inf"))
    distances, indices = floor_distances.topk(k, largest=False)
    weights = (distances.sqrt() + 1e-4).pow(-float(settings["distance_power"]))
    weights = weights / weights.sum(1, keepdim=True)
    nearest_xy = (weights[..., None] * train_xy[indices]).sum(1)
    # Both heads use the predicted floor, never the true query label.
    neural_xy = output["conditional_xy"][torch.arange(len(signal), device=signal.device), predicted]
    alpha = float(settings["neural_coordinate_weight"])
    return {"logits": probabilities.clamp_min(1e-12).log(),
            "xy": alpha * neural_xy + (1 - alpha) * nearest_xy}


class RefinedKANMixer(nn.Module):
    def __init__(self, backbone, payload):
        super().__init__()
        self.backbone = backbone
        self.settings = payload["settings"]
        if payload.get("fit_scope") not in ("internal_train_only", "development_after_validation_selection"):
            raise ValueError("Fingerprint refinement requires an internal-train-only catalogue.")
        if not 0 < self.settings["neural_coordinate_weight"] <= 1:
            raise ValueError("Hybrid must retain a positive neural coordinate contribution.")
        if not 0 <= self.settings["neighbor_class_weight"] <= 1:
            raise ValueError("Invalid neighbor classification weight.")
        if self.settings["neighbors"] < 1 or self.settings["distance_power"] < 0:
            raise ValueError("Invalid neighbor decoding settings.")
        signal, xy, labels = payload["signal"], payload["xy"], payload["class_ids"]
        if (signal.ndim != 2 or signal.shape[1] != 520 or xy.shape != (len(signal), 2)
                or labels.shape != (len(signal),) or len(signal) == 0
                or not torch.isfinite(signal).all() or not torch.isfinite(xy).all()
                or (labels < 0).any() or (labels >= backbone.classes).any()
                or any(not (labels == c).any() for c in range(backbone.classes))):
            raise ValueError("Invalid train fingerprint catalogue.")
        self.register_buffer("train_signal", signal)
        self.register_buffer("train_xy", xy)
        self.register_buffer("train_class_ids", labels)

    def forward(self, features):
        return refine_predictions(self.backbone(features), features,
            (self.train_signal, self.train_xy, self.train_class_ids), self.settings)
