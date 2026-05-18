#!/usr/bin/env python3
"""Train Ligandformer on a labeled molecular CSV file."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ligandformer.dataset import MoleculeGraphDataset, collect_smiles
from ligandformer.featurization import AtomVocabulary
from ligandformer.metrics import binary_classification_metrics
from ligandformer.utils import build_model, get_device, load_yaml, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ligandformer for binary molecular classification.")
    parser.add_argument("--train_csv", required=True, help="Training CSV with SMILES and label columns.")
    parser.add_argument("--test_csv", default="", help="Optional test/validation CSV.")
    parser.add_argument("--output_dir", default="runs/ligandformer", help="Directory for checkpoints and reports.")
    parser.add_argument("--config", default="configs/ligandformer_classification.yaml", help="YAML configuration file.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Used only when --test_csv is omitted.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_add_hydrogens", action="store_true")
    parser.add_argument(
        "--no_balanced_sampling",
        action="store_true",
        help="Disable 1:1 positive/negative resampling during training.",
    )
    return parser.parse_args()


def build_balanced_sampler(graphs, seed: int) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that balances positive and negative labels to 1:1."""

    labels = np.array([float(g.y.item()) for g in graphs])
    if labels.size == 0:
        raise ValueError("Cannot build a balanced sampler on an empty dataset.")
    positives = float((labels > 0.5).sum())
    negatives = float((labels <= 0.5).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Balanced sampling requires both positive and negative samples.")
    pos_weight = 0.5 / positives
    neg_weight = 0.5 / negatives
    weights = np.where(labels > 0.5, pos_weight, neg_weight)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def split_dataset(dataset, val_fraction: float, seed: int):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = max(1, int(round(len(indices) * val_fraction)))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx]


def run_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = loss_fn(logits.view(-1), batch.y.view(-1).float())
        loss.backward()
        optimizer.step()
        batch_size = batch.y.numel()
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all = []
    labels_all = []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        loss = loss_fn(logits.view(-1), batch.y.view(-1).float())
        batch_size = batch.y.numel()
        total_loss += loss.item() * batch_size
        total_count += batch_size
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(batch.y.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all).reshape(-1)
    labels_np = np.concatenate(labels_all).reshape(-1)
    metrics = binary_classification_metrics(logits_np, labels_np)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 100))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 128))
    learning_rate = args.learning_rate if args.learning_rate is not None else float(cfg.get("learning_rate", 0.001))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(cfg.get("weight_decay", 0.0001))
    patience = args.early_stop_patience if args.early_stop_patience is not None else int(cfg.get("early_stop_patience", 50))
    add_hydrogens = bool(cfg.get("add_hydrogens", True)) and not args.no_add_hydrogens
    monitor_metric = str(cfg.get("monitor_metric", "auroc"))
    balanced_sampling = bool(cfg.get("balanced_sampling", True)) and not args.no_balanced_sampling
    model_config = dict(cfg.get("model", {}))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    vocab_paths = [args.train_csv] + ([args.test_csv] if args.test_csv else [])
    atom_vocab = AtomVocabulary.from_smiles(collect_smiles(vocab_paths), add_hydrogens=add_hydrogens)

    full_train_dataset = MoleculeGraphDataset(args.train_csv, atom_vocab, add_hydrogens=add_hydrogens)
    if args.test_csv:
        train_dataset = full_train_dataset
        val_dataset = MoleculeGraphDataset(args.test_csv, atom_vocab, add_hydrogens=add_hydrogens)
    else:
        train_dataset, val_dataset = split_dataset(full_train_dataset, args.val_fraction, seed)

    if balanced_sampling:
        sampler = build_balanced_sampler(train_dataset, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=len(train_dataset) > batch_size,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > batch_size,
            num_workers=args.num_workers,
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(len(atom_vocab), model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_score = None
    stale_epochs = 0
    history = []
    best_path = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        score = val_metrics.get(monitor_metric, -val_metrics["loss"])
        if score is None or (isinstance(score, float) and math.isnan(score)):
            score = -val_metrics["loss"]

        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                atom_vocab,
                model_config,
                {
                    "epoch": epoch,
                    "seed": seed,
                    "train_csv": str(args.train_csv),
                    "test_csv": str(args.test_csv),
                    "add_hydrogens": add_hydrogens,
                    "validation_metrics": val_metrics,
                },
            )
        else:
            stale_epochs += 1

        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_auroc={val_metrics['auroc']:.5f} "
            f"val_auprc={val_metrics['auprc']:.5f}"
        )

        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best {monitor_metric}={best_score:.5f}.")
            break

    save_json({"history": history}, output_dir / "history.json")
    save_json({"best_checkpoint": str(best_path), "best_score": best_score}, output_dir / "summary.json")
    print(f"Best checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
