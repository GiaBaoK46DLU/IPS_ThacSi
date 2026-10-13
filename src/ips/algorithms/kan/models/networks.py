"""Biến thể RSSI theo chặng 4, không phải tái lập benchmark ảnh của bài báo."""
from __future__ import annotations

import torch
from torch import nn

from .vendor.efficient_kan import KANLinear


class PositionHeads(nn.Module):
    def __init__(self, width: int, classes: int):
        super().__init__()
        self.classifier = nn.Linear(width, classes)
        self.coordinates = nn.Linear(width, 2)

    def forward(self, representation):
        # CrossEntropyLoss nhận logits; tọa độ đang trong hệ chuẩn hóa train.
        return {"logits": self.classifier(representation),
                "xy": self.coordinates(representation)}


def mixing_branch(width: int, hidden: int, dropout: float, kan: dict):
    options = {key: kan[key] for key in ["grid_size", "spline_order", "scale_noise", "scale_base",
                "scale_spline", "enable_standalone_scale_spline", "grid_eps", "grid_range"]}
    # KANLinear đã có base SiLU và spline học được. Không chèn GELU giữa hai KAN.
    return nn.Sequential(KANLinear(width, hidden, **options), nn.Dropout(dropout),
                         KANLinear(hidden, width, **options), nn.Dropout(dropout))


class MixerBlock(nn.Module):
    def __init__(self, tokens: int, channels: int, token_hidden: int,
                 channel_hidden: int, dropout: float, kan: dict):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_mixer = mixing_branch(tokens, token_hidden, dropout, kan)
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mixer = mixing_branch(channels, channel_hidden, dropout, kan)

    def forward(self, x):
        # [B,T,C] -> [B,C,T]: mỗi channel dùng chung phép trộn T tokens.
        token_update = self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + token_update
        # [B,T,C]: mỗi token dùng chung phép trộn C channels.
        return x + self.channel_mixer(self.channel_norm(x))


class RSSIMixer(nn.Module):
    def __init__(self, settings: dict, classes: int, kan: dict):
        super().__init__()
        self.tokens = settings["tokens"]
        self.features_per_token = settings["features_per_token"]
        if self.tokens * self.features_per_token != 1040:
            raise ValueError("Tokenization must retain all 520 AP × 2 features.")
        width = settings["channels"]
        self.embedding = nn.Linear(self.features_per_token, width)
        self.blocks = nn.Sequential(*[MixerBlock(self.tokens, width, settings["token_hidden"],
                settings["channel_hidden"], settings["dropout"], kan) for _ in range(settings["blocks"])])
        self.final_norm = nn.LayerNorm(width)
        self.heads = PositionHeads(width, classes)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (520, 2):
            raise ValueError("Expected input [B,520,2] in AP order (z,mask).")
        patches = features.reshape(features.shape[0], self.tokens, self.features_per_token)
        x = self.blocks(self.embedding(patches))
        return self.heads(self.final_norm(x).mean(dim=1))


def build_model(config: dict, classes: int = 13) -> nn.Module:
    name = config["model"]
    if name in ("mlp", "mlp_mixer"):
        raise ValueError("Personal MLP baselines are excluded from this shared project; choose kan_mixer.")
    if name == "kan_mixer":
        if config["kan"]["base_activation"] != "SiLU":
            raise ValueError("Pinned IPS protocol uses the efficient-kan SiLU base activation.")
        if config["kan"]["update_grid"] or config["kan"]["extra_regularization_loss_coefficient"] != 0:
            raise ValueError("Protocol v1 fixes grid and uses no extra KAN regularization loss.")
        variant = config.get("positioning", {}).get("variant", "regression_v1")
        if variant == "reference":
            from .reference import ReferenceKANMixer
            return ReferenceKANMixer(config, classes)
        if variant != "regression_v1":
            raise ValueError(f"Unknown KAN positioning variant: {variant}")
        return RSSIMixer(config["mixer"], classes, config["kan"])
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
