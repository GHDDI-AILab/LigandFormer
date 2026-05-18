#!/usr/bin/env python3
"""Run Ligandformer prediction on a molecular CSV file."""

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
from ligandformer.featurization import atom_attention_scores
from ligandformer.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict molecular property probabilities with Ligandformer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_attention", action="store_true")
    parser.add_argument("--attention_dir", default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, atom_vocab, checkpoint = load_checkpoint(args.checkpoint, device=device, viz_att=args.save_attention)
    add_hydrogens = bool(checkpoint.get("metadata", {}).get("add_hydrogens", True))
    dataset = MoleculeGraphDataset(
        args.input_csv,
        atom_vocab,
        add_hydrogens=add_hydrogens,
        require_labels=False,
    )

    batch_size = 1 if args.save_attention else args.batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    attention_dir = Path(args.attention_dir) if args.attention_dir else Path(args.output_csv).with_suffix("").parent / "attention"
    if args.save_attention:
        attention_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    row_index = 0
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        if args.save_attention:
            logits, _, attention_maps = model(batch)
        else:
            logits, _ = model(batch)
            attention_maps = None

        logits_np = logits.detach().cpu().numpy().reshape(-1)
        probabilities = 1.0 / (1.0 + np.exp(-logits_np))
        labels = batch.y.detach().cpu().numpy().reshape(-1) if getattr(batch, "y", None) is not None else [None] * len(logits_np)

        for i, logit in enumerate(logits_np):
            rows.append(
                {
                    "smiles": batch.smiles[i],
                    "label": labels[i] if labels[i] is not None else "",
                    "logit": logit,
                    "probability": probabilities[i],
                    "prediction": int(probabilities[i] >= 0.5),
                }
            )

        if args.save_attention and attention_maps is not None:
            scores = atom_attention_scores([attention.detach().cpu() for attention in attention_maps])
            arrays = {f"block_{i}": attention.detach().cpu().numpy() for i, attention in enumerate(attention_maps)}
            if scores is not None:
                arrays["integrated_atom_scores"] = scores.numpy()
            np.savez(attention_dir / f"row_{row_index}.npz", **arrays)
        row_index += len(logits_np)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}")
    if args.save_attention:
        print(f"Attention arrays written to {attention_dir}")


if __name__ == "__main__":
    main()
