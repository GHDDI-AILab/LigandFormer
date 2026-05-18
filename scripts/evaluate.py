#!/usr/bin/env python3
"""Evaluate a trained Ligandformer checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ligandformer.dataset import MoleculeGraphDataset
from ligandformer.metrics import binary_classification_metrics
from ligandformer.utils import get_device, load_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Ligandformer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits_all = []
    labels_all = []
    smiles_all = []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(batch.y.detach().cpu().numpy())
        smiles_all.extend(batch.smiles)
    return (
        np.concatenate(logits_all).reshape(-1),
        np.concatenate(labels_all).reshape(-1),
        smiles_all,
    )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, atom_vocab, checkpoint = load_checkpoint(args.checkpoint, device=device)
    add_hydrogens = bool(checkpoint.get("metadata", {}).get("add_hydrogens", True))
    dataset = MoleculeGraphDataset(args.test_csv, atom_vocab, add_hydrogens=add_hydrogens)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logits, labels, smiles = predict_logits(model, loader, device)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    metrics = binary_classification_metrics(logits, labels)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "metrics.json")
    pd.DataFrame(
        {
            "smiles": smiles,
            "label": labels.astype(int),
            "logit": logits,
            "probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
        }
    ).to_csv(output_dir / "predictions.csv", index=False)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Predictions written to {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
