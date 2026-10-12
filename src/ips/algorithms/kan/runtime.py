"""Seed, device, cấu hình và checkpoint có metadata tự chứa."""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import sklearn
import torch

from ...data import ROOT, json_bytes, sha256
from .models import build_model
from .paths import resolve_config_path


def configure_runtime(seed: int, device: str, threads: int = 4):
    if threads < 1:
        raise ValueError("threads must be positive.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Use --device cpu or install compatible CUDA PyTorch.")
    return torch.device(device)


def read_config(path: Path) -> dict:
    path = resolve_config_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "split_dir", "mixer", "kan", "loss", "training"}
    if not required.issubset(config):
        raise ValueError(f"Config lacks: {sorted(required - set(config))}")
    train = config["training"]
    if (train["batch_size"] < 1 or train["max_epochs"] < 1 or train["patience"] < 1 or
        train["learning_rate"] <= 0 or train["weight_decay"] < 0 or train["gradient_clip_l2"] <= 0 or
        config["loss"]["lambda_xy"] < 0 or config["loss"]["smooth_l1_beta"] <= 0):
        raise ValueError("Invalid training/loss hyperparameters.")
    if config["loss"].get("reference_sigma_normalized", 0) < 0:
        raise ValueError("Reference spatial smoothing radius must be nonnegative.")
    if train.get("scheduler") not in (None, "cosine"):
        raise ValueError("Supported scheduler: cosine (or omit for constant learning rate).")
    return config


def save_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def software_info(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "platform": platform.platform(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "threads": torch.get_num_threads()}


def code_hashes():
    return {path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted((ROOT / "src" / "ips").rglob("*.py"))}


def load_checkpoint(path, device):
    # Payload chỉ gồm state_dict, tensor và kiểu primitive: weights_only=True.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    model = build_model(checkpoint["config"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
