"""Utility functions shared by training, evaluation, and prediction scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from .featurization import AtomVocabulary, node_feature_dim
from .model import LigandFormer


DEFAULT_MODEL_CONFIG = {
    "block_num": 3,
    "embedding_dim": 75,
    "conv_hidden_dim": 256,
    "classifier_hidden_dim": 256,
    "output_dim": 1,
    "aggregation_methods": ["max", "sum"],
    "multiple_aggregation_merge_method": "sum",
    "node_feature_update_method": "cat",
    "readout_methods": "mean",
    "pyramid_feature": True,
    "att_num_heads": 1,
    "dropout": 0.1,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def build_model(vocab_size: int, config: Optional[Dict[str, Any]] = None, viz_att: bool = False) -> LigandFormer:
    model_config = dict(DEFAULT_MODEL_CONFIG)
    if config:
        model_config.update(config)
    model_config["viz_att"] = viz_att
    feat_dim = node_feature_dim(vocab_size)
    return LigandFormer(node_feature_dim=feat_dim, **model_config)


def save_checkpoint(
    path: str | Path,
    model: LigandFormer,
    atom_vocab: AtomVocabulary,
    model_config: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "atom_vocab": atom_vocab.to_dict(),
            "model_config": model_config,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device, viz_att: bool = False):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    atom_vocab = AtomVocabulary.from_dict(checkpoint["atom_vocab"])
    model_config = dict(checkpoint.get("model_config", {}))
    model = build_model(len(atom_vocab), model_config, viz_att=viz_att)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing}, unexpected={unexpected}")
    model.to(device)
    return model, atom_vocab, checkpoint
