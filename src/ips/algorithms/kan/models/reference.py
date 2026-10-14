"""Experimental KAN-Mixer with train-only reference-point supervision.

This is an IPS-specific variant, not the image architecture in Canuto et al.
Only RSSI features enter forward(). Reference coordinates are training targets
stored in the selected checkpoint; no query location metadata is consumed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .networks import MixerBlock


def fit_reference_points(config, dataset, *, final_refit=False):
    """Fit the catalogue on the supplied training dataset, never validation."""
    settings = config.get("positioning", {})
    if settings.get("variant") != "reference":
        return
    expected = "development" if final_refit else "train"
    if getattr(dataset, "split_name", None) != expected:
        raise ValueError("Reference fitting requires an explicitly identified train split.")
    keys = np.column_stack([dataset.classes.numpy(), dataset.xy_original])
    points = np.unique(keys, axis=0)
    xy = dataset.xy.numpy()
    # Reconstruct the existing train-only coordinate transform from paired rows.
    # Use original coordinates for grouping (float32 rounded keys can collide).
    lookup = {(int(c), float(x), float(y)): normalized.tolist()
              for (c, x, y), normalized in zip(keys, xy)}
    settings["references"] = [
        {"class_id": int(c), "xy": lookup[(int(c), float(x), float(y))]}
        for c, x, y in points]
    settings["fit_scope"] = "development_after_validation_selection" if final_refit else "internal_train_only"


class ReferenceKANMixer(nn.Module):
    def __init__(self, config, classes):
        super().__init__()
        mixer, kan, settings = config["mixer"], config["kan"], config["positioning"]
        references = settings.get("references", [])
        if not references or settings.get("fit_scope") not in ("internal_train_only", "development_after_validation_selection"):
            raise ValueError("Reference catalogue must be fitted on internal train before model creation.")
        coordinates = torch.tensor([p["xy"] for p in references], dtype=torch.float32)
        labels = torch.tensor([p["class_id"] for p in references], dtype=torch.long)
        if (coordinates.shape != (len(references), 2) or not torch.isfinite(coordinates).all()
                or (labels < 0).any() or (labels >= classes).any()
                or any(not (labels == c).any() for c in range(classes))):
            raise ValueError("Invalid reference catalogue or missing building/floor class.")
        self.register_buffer("reference_xy", coordinates)
        self.register_buffer("reference_classes", labels)
        # Pad floor catalogues once on CPU. Avoid 13 boolean CUDA indexing
        # synchronizations in every training batch; buffers are derived state.
        counts = [int((labels == c).sum()) for c in range(classes)]
        indices = torch.zeros(classes, max(counts), dtype=torch.long)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        for c, count in enumerate(counts):
            indices[c, :count] = torch.where(labels == c)[0]
            valid[c, :count] = True
        self.register_buffer("catalogue_indices", indices, persistent=False)
        self.register_buffer("catalogue_valid", valid, persistent=False)
        self.classes, self.settings = classes, settings
        if (not 0 <= settings.get("ap_dropout", 0) < 1 or settings.get("mask_scale", 0) < 0
                or settings.get("rssi_noise_db", 0) < 0 or settings.get("device_shift_db", 0) < 0):
            raise ValueError("Invalid RSSI augmentation or mask scale.")
        self.tokens, self.features_per_token = mixer["tokens"], mixer["features_per_token"]
        if self.tokens * self.features_per_token != 1040:
            raise ValueError("Tokenization must retain 520 AP × 2 features.")
        width = mixer["channels"]
        self.embedding = nn.Linear(self.features_per_token, width)
        self.blocks = nn.Sequential(*[MixerBlock(self.tokens, width, mixer["token_hidden"],
            mixer["channel_hidden"], mixer["dropout"], kan) for _ in range(mixer["blocks"])])
        self.final_norm = nn.LayerNorm(width)
        self.pooling = settings.get("pooling", "flatten")
        if self.pooling not in ("flatten", "mean"):
            raise ValueError("Reference pooling must be flatten or mean.")
        head_input = self.tokens * width if self.pooling == "flatten" else width
        hidden = settings.get("head_width", 256)
        self.head = nn.Sequential(nn.Linear(head_input, hidden), nn.GELU(),
                                  nn.Dropout(mixer["dropout"]), nn.Linear(hidden, len(references)))
        if settings.get("temperature", 1.0) <= 0 or settings.get("top_k", 5) < 1:
            raise ValueError("Reference temperature/top_k must be positive.")

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (520, 2):
            raise ValueError("Expected input [B,520,2] in AP order (z,mask).")
        signal, mask = features[..., 0], features[..., 1]
        if self.training:
            noise = float(self.settings.get("rssi_noise_db", 0)) / 104
            shift = float(self.settings.get("device_shift_db", 0)) / 104
            signal = signal + noise * torch.randn_like(signal)
            signal = signal + shift * torch.randn((len(signal), 1), device=signal.device)
            drop = float(self.settings.get("ap_dropout", 0))
            if drop:
                mask = mask * (torch.rand_like(mask) >= drop)
            signal = signal.clamp(0, 1) * mask
        transformed = torch.stack([signal, mask * self.settings.get("mask_scale", 0.0)], -1)
        patches = transformed.reshape(len(features), self.tokens, self.features_per_token)
        x = self.final_norm(self.blocks(self.embedding(patches)))
        representation = x.flatten(1) if self.pooling == "flatten" else x.mean(1)
        reference_logits = self.head(representation)
        temperature = self.settings.get("temperature", 1.0)
        scores = reference_logits[:, self.catalogue_indices].masked_fill(
            ~self.catalogue_valid[None], float("-inf"))
        positions = self.reference_xy[self.catalogue_indices]
        logits = torch.logsumexp(scores, dim=-1)
        training_temperature = self.settings.get("training_temperature", temperature)
        dense_conditional = torch.einsum("bcj,cjd->bcd", (scores / training_temperature).softmax(-1), positions)
        k = min(self.settings.get("top_k", 5), scores.shape[-1])
        top_scores, indices = scores.topk(k, dim=-1)
        expanded_positions = positions[None].expand(len(features), -1, -1, -1)
        top_positions = expanded_positions.gather(2, indices[..., None].expand(-1, -1, -1, 2))
        conditional_xy = ((top_scores / temperature).softmax(-1)[..., None] * top_positions).sum(2)
        predicted_class = logits.argmax(1)
        xy = conditional_xy[torch.arange(len(features), device=features.device), predicted_class]
        return {"logits": logits, "xy": xy, "reference_logits": reference_logits,
                "reference_xy": self.reference_xy, "reference_classes": self.reference_classes,
                "dense_conditional_xy": dense_conditional, "conditional_xy": conditional_xy}
