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
from .vendor.efficient_kan import KANLinear


def observed_signal_rank(signal, mask):
    """Average tied ranks among observed APs, mapped to [-1,1].

    AP identity/order is retained. This is a per-query, non-fitted feature;
    missing APs and queries with <=1 observed AP receive zero ranks.
    """
    values = signal.contiguous()
    ordered = signal.masked_fill(~mask.bool(), float('inf')).sort(-1).values.contiguous()
    left = torch.searchsorted(ordered, values, right=False)
    right = torch.searchsorted(ordered, values, right=True)
    average_rank = (left + right - 1).to(signal.dtype) / 2
    count = mask.sum(-1, keepdim=True)
    rank = 2 * average_rank / (count - 1).clamp_min(1) - 1
    return torch.where(count > 1, rank * mask, torch.zeros_like(signal))


def represent_signal(signal, mask, settings):
    """Per-fingerprint representations; no fitted held-out statistics.

    The default is byte-for-byte the historical two-channel stack. Relative
    RSSI subtracts the observed AP mean of the same query, with missing APs
    excluded, then rescales by 0.2 (=20.8 dB). It adds no location metadata.
    """
    mode = settings.get("input_mode", "absolute_mask")
    if mode == "absolute_mask":
        return torch.stack([signal, mask * settings.get("mask_scale", 0.0)], -1)
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    mean = (signal * mask).sum(-1, keepdim=True) / count
    relative = (signal - mean) * mask / 0.2
    if mode == "absolute_relative_mask":
        return torch.stack([signal, relative, mask * settings.get("mask_scale", 1.0)], -1)
    if mode == "relative_mask":
        return torch.stack([relative, mask * settings.get("mask_scale", 1.0)], -1)
    if mode == "relative_rank_mask":
        return torch.stack([relative, observed_signal_rank(signal, mask),
                            mask * settings.get("mask_scale", 1.0)], -1)
    raise ValueError(f"Unknown RSSI input_mode: {mode}")


def fit_reference_points(config, dataset, *, final_refit=False):
    """Fit the catalogue on the supplied training dataset, never validation."""
    settings = config.get("positioning", {})
    if settings.get("variant") != "reference":
        return
    expected = "development" if final_refit else "train"
    if getattr(dataset, "split_name", None) != expected:
        raise ValueError("Reference fitting requires an explicitly identified train split.")
    keys = np.column_stack([dataset.classes.numpy(), dataset.xy_original])
    if settings.get('head_type', 'point_classifier') != 'point_classifier':
        raise ValueError('Only the KAN-Mixer point_classifier head is supported; fingerprint memory is excluded.')
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
    from ..v4_methods import fit_ap_groups
    fit_ap_groups(config, dataset, final_refit=final_refit)


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
        if settings.get('ap_grouping', 'source_order') != 'source_order':
            from ..v4_methods import validate_permutation
            validate_permutation(settings.get('ap_permutation'))
            if settings.get('ap_permutation_fit_scope') != 'internal_train_only':
                raise ValueError('AP grouping must be fitted on internal training data.')
            self.register_buffer('ap_permutation', torch.tensor(settings['ap_permutation']), persistent=False)
        if (not 0 <= settings.get("ap_dropout", 0) < 1 or settings.get("mask_scale", 0) < 0
                or settings.get("rssi_noise_db", 0) < 0 or settings.get("device_shift_db", 0) < 0):
            raise ValueError("Invalid RSSI augmentation or mask scale.")
        self.tokens, self.features_per_token = mixer["tokens"], mixer["features_per_token"]
        input_mode = settings.get("input_mode", "absolute_mask")
        if input_mode not in ("absolute_mask", "absolute_relative_mask", "relative_mask", "relative_rank_mask"):
            raise ValueError(f"Unknown RSSI input_mode: {input_mode}")
        channels_per_ap = 3 if input_mode in ("absolute_relative_mask", "relative_rank_mask") else 2
        if self.tokens * self.features_per_token != 520 * channels_per_ap:
            raise ValueError(f"Tokenization must retain 520 AP × {channels_per_ap} features.")
        width = mixer["channels"]
        self.embedding = nn.Linear(self.features_per_token, width)
        self.blocks = nn.Sequential(*[MixerBlock(self.tokens, width, mixer["token_hidden"],
            mixer["channel_hidden"], mixer["dropout"], kan) for _ in range(mixer["blocks"])])
        self.final_norm = nn.LayerNorm(width)
        self.pooling = settings.get("pooling", "flatten")
        if self.pooling not in ("flatten", "mean", "learned"):
            raise ValueError("Reference pooling must be flatten, mean or learned.")
        self.token_pool = (nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 1))
                           if self.pooling == "learned" else None)
        head_input = self.tokens * width if self.pooling == "flatten" else width
        hidden = settings.get("head_width", 256)
        head_transform = settings.get("head_transform", "linear_gelu")
        if head_transform == "linear_gelu":
            # Historical initialization order and state_dict keys stay identical.
            self.head = nn.Sequential(nn.Linear(head_input, hidden), nn.GELU(),
                                      nn.Dropout(mixer["dropout"]), nn.Linear(hidden, len(references)))
        elif head_transform == "kan":
            options = {key: kan[key] for key in ["grid_size", "spline_order", "scale_noise", "scale_base",
                "scale_spline", "enable_standalone_scale_spline", "grid_eps", "grid_range"]}
            self.head = nn.Sequential(KANLinear(head_input, hidden, **options), nn.Dropout(mixer["dropout"]),
                                      KANLinear(hidden, len(references), **options))
        else:
            raise ValueError(f"Unknown KAN-Mixer head_transform: {head_transform}")
        self.auxiliary_classifier = (nn.Linear(head_input, classes)
                                     if settings.get("auxiliary_classifier", False) else None)
        self.residual_scale = float(settings.get('coordinate_residual_scale', 0.))
        if not np.isfinite(self.residual_scale) or not 0 <= self.residual_scale <= .25:
            raise ValueError('coordinate_residual_scale must be finite within [0,0.25].')
        self.coordinate_residual = None
        if self.residual_scale:
            residual_width = settings.get('coordinate_residual_width', 32)
            if type(residual_width) is not int or residual_width < 1:
                raise ValueError('coordinate_residual_width must be a positive integer.')
            options = {key: kan[key] for key in ['grid_size', 'spline_order', 'scale_noise', 'scale_base',
                'scale_spline', 'enable_standalone_scale_spline', 'grid_eps', 'grid_range']}
            self.coordinate_residual = nn.Sequential(KANLinear(head_input, residual_width, **options),
                KANLinear(residual_width, classes * 2, **options))
            # Initially reproduce the point decoder. Keep spline_scaler nonzero
            # so the last spline weights can learn on their first update.
            with torch.no_grad():
                self.coordinate_residual[-1].base_weight.zero_()
                self.coordinate_residual[-1].spline_weight.zero_()
        if settings.get("temperature", 1.0) <= 0 or settings.get("top_k", 5) < 1:
            raise ValueError("Reference temperature/top_k must be positive.")
        if settings.get("training_temperature", 1.0) <= 0 or settings.get("training_top_k", 1) < 1:
            raise ValueError("Training temperature/top_k must be positive.")
        if settings.get("inference_shifts_db"):
            raise ValueError("Use one KAN-Mixer forward pass; multi-view averaging is excluded.")

    def forward(self, features):
        return self._forward_single(features)

    def _forward_single(self, features):
        representation = self.encode_features(features)
        reference_logits = self.head(representation)
        auxiliary_logits = (self.auxiliary_classifier(representation)
                            if self.auxiliary_classifier is not None else None)
        residual = (self.residual_scale * self.coordinate_residual(representation).reshape(-1, self.classes, 2).tanh()
                    if self.coordinate_residual is not None else None)
        return self._decode(reference_logits, auxiliary_logits, residual)

    def encode_features(self, features):
        """Shared RSSI-to-representation route; preserves historical operations."""
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
        transformed = represent_signal(signal, mask, self.settings)
        if hasattr(self, 'ap_permutation'):
            transformed = transformed[:, self.ap_permutation]
        patches = transformed.reshape(len(features), self.tokens, self.features_per_token)
        x = self.final_norm(self.blocks(self.embedding(patches)))
        if self.pooling == "flatten":
            representation = x.flatten(1)
        elif self.pooling == "mean":
            representation = x.mean(1)
        else:
            representation = (self.token_pool(x).softmax(1) * x).sum(1)
        return representation

    def _decode(self, reference_logits, auxiliary_logits=None, coordinate_residual=None):
        temperature = self.settings.get("temperature", 1.0)
        scores = reference_logits[:, self.catalogue_indices].masked_fill(
            ~self.catalogue_valid[None], float("-inf"))
        positions = self.reference_xy[self.catalogue_indices]
        logits = torch.logsumexp(scores, dim=-1)
        if auxiliary_logits is not None:
            logits = logits + self.settings.get("class_logit_weight", 1.0) * auxiliary_logits
        training_temperature = self.settings.get("training_temperature", temperature)
        dense_conditional = torch.einsum("bcj,cjd->bcd", (scores / training_temperature).softmax(-1), positions)
        decoder_k = self.settings.get("top_k", 5)
        if self.training:
            decoder_k = self.settings.get("training_top_k", decoder_k)
            temperature = training_temperature
        k = min(decoder_k, scores.shape[-1])
        top_scores, indices = scores.topk(k, dim=-1)
        expanded_positions = positions[None].expand(len(reference_logits), -1, -1, -1)
        top_positions = expanded_positions.gather(2, indices[..., None].expand(-1, -1, -1, 2))
        conditional_xy = ((top_scores / temperature).softmax(-1)[..., None] * top_positions).sum(2)
        if coordinate_residual is not None:
            dense_conditional = dense_conditional + coordinate_residual
            conditional_xy = conditional_xy + coordinate_residual
        predicted_class = logits.argmax(1)
        xy = conditional_xy[torch.arange(len(reference_logits), device=reference_logits.device), predicted_class]
        result = {"logits": logits, "xy": xy, "reference_logits": reference_logits,
                "reference_xy": self.reference_xy, "reference_classes": self.reference_classes,
                "dense_conditional_xy": dense_conditional, "conditional_xy": conditional_xy}      
        if auxiliary_logits is not None:
            result["auxiliary_logits"] = auxiliary_logits
        if coordinate_residual is not None:
            result['coordinate_residual'] = coordinate_residual
        return result
