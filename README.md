# Ligandformer

[![Paper](https://img.shields.io/badge/Paper-Ligandformer.pdf-red)](Ligandformer.pdf)
(https://arxiv.org/abs/2202.10873)

Ligandformer is a multi-layer self-attention graph neural network for molecular property prediction with robust atom-level interpretation. This repository contains:

- `Ligandformer.tex` / `Ligandformer.pdf` — the manuscript.
- `ligandformer/` — the Python package implementing the model, featurization, data loading, and metrics.
- `scripts/` — command line tools for training, evaluation, and prediction with block-level attention export.
- `configs/` — default training and model hyperparameters.
- `examples/` — ready-to-run shell scripts for the three datasets reported in the paper.
- `datasets/` — the Aqueous Solubility, Caco-2, and Ames Mutagenesis splits used in the manuscript experiments.

Official repository: [https://github.com/GHDDI-AILab/LigandFormer](https://github.com/GHDDI-AILab/LigandFormer)

## Repository Layout

```text
LigandFormer/
├── Ligandformer.tex              Manuscript source
├── Ligandformer.pdf              Compiled paper
├── references.bib                Bibliography
├── arxiv.sty, orcid.pdf          LaTeX template assets
├── ligandformer_framework.png    Figure 1: architecture
├── att_viz.png                   Figure 2: attention visualization
├── diff_rounds.png               Figure 3: robustness across seeds
├── README.md
├── License.txt
├── requirements.txt
├── configs/
│   └── ligandformer_classification.yaml
├── ligandformer/
│   ├── __init__.py
│   ├── model.py                  LigandFormer model + HAGConv + self-attention blocks
│   ├── featurization.py          7-attribute atom featurization (Fig. 1c)
│   ├── dataset.py                CSV → PyG graphs
│   ├── metrics.py                Binary classification metrics
│   └── utils.py                  Model build / save / load / seed helpers
├── scripts/
│   ├── train.py                  Training with 1:1 balanced resampling
│   ├── evaluate.py               Metric reporting on held-out CSV
│   └── predict.py                Probability + attention map export
├── examples/
│   ├── train_caco2.sh
│   ├── train_ames.sh
│   └── train_logp.sh
└── datasets/
    ├── Caco2-train6862.csv, Caco2-test762.csv
    ├── T-ames-cleanedSMILES_train6517.csv, T-ames_cleanedSMILES_test761.csv
    └── wat_logP_csvAll_cols_labeled_train1179.csv, wat_logP_csvAll_cols_labeled_test131.csv
```

## Installation

Python 3.8+ is required. Install the scientific Python stack, RDKit, PyTorch, and PyTorch Geometric:

```bash
python3 -m pip install -r requirements.txt
```

`torch-geometric` and `torch-scatter` must match your PyTorch / CUDA build — follow the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) when installing on GPU machines.

## Data Format

Input CSV files should contain a SMILES column and, for training/evaluation, a binary label column. The loaders accept common column names automatically:

- SMILES columns: `cleaned_smiles`, `Cleaned_SMILES`, `smiles`, `SMILES`
- Label columns: `label`, `Label`, `y`, `Y`

The `datasets/` directory ships with the three splits reported in the paper:

| Property | Train | Test | Files |
| --- | --- | --- | --- |
| Aqueous Solubility (logS) | 1,179 | 131 | `wat_logP_csvAll_cols_labeled_{train1179,test131}.csv` |
| Caco-2 Cell Permeability | 6,862 | 762 | `Caco2-{train6862,test762}.csv` |
| Ames Mutagenesis | 6,517 | 761 | `T-ames-cleanedSMILES_train6517.csv`, `T-ames_cleanedSMILES_test761.csv` |

## Training

```bash
python3 scripts/train.py \
  --train_csv datasets/Caco2-train6862.csv \
  --test_csv datasets/Caco2-test762.csv \
  --output_dir runs/caco2 \
  --config configs/ligandformer_classification.yaml
```

The default configuration follows the manuscript: 3 self-attention blocks, `batch_size=256`, Adam with `lr=1e-3` and `weight_decay=1e-4`, mean read-out, and 1:1 positive/negative resampling. Disable balanced sampling with `--no_balanced_sampling`.

Convenience wrappers for the three reported tasks:

```bash
bash examples/train_caco2.sh
bash examples/train_ames.sh
bash examples/train_logp.sh
```

Each run writes `best.pt`, `history.json`, and `summary.json` into the output directory.

## Evaluation

```bash
python3 scripts/evaluate.py \
  --checkpoint runs/caco2/best.pt \
  --test_csv datasets/Caco2-test762.csv \
  --output_dir runs/caco2_eval
```

The evaluator writes `metrics.json` and `predictions.csv`.

## Prediction and Attention Maps

```bash
python3 scripts/predict.py \
  --checkpoint runs/caco2/best.pt \
  --input_csv datasets/Caco2-test762.csv \
  --output_csv runs/caco2_predictions.csv \
  --save_attention
```

With `--save_attention`, each molecule is processed individually and one `.npz` per row is written containing:

- `block_{i}` — the `(n_atoms, n_atoms)` self-attention matrix from block *i*.
- `integrated_atom_scores` — the row-averaged attention scores integrated across all blocks (Figure 2).

## Reproducing Manuscript Numbers

The AUROC numbers reported in Table 1 (Aqueous Solubility 0.98, Caco-2 0.89, Ames 0.92) can be reproduced by running the three `examples/train_*.sh` scripts with the default config. Expect small seed-level variation; Section 3.3 and Figure 3 document how integrated attention maps remain stable across training rounds.

## Building The Manuscript

```bash
latexmk -pdf Ligandformer.tex
```

Requires a TeX Live distribution with `fancyhdr`, `units` (nicefrac), `microtype`, `natbib`, `doi`, `hyperref`, `booktabs`, and `graphicx` installed.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{guo2026ligandformergraphneuralnetwork,
      title={Ligandformer: A Graph Neural Network for Predicting Compound Property with Robust Interpretation}, 
      author={Jinjiang Guo and Qi Liu and Han Guo and Xi Lu},
      year={2026},
      eprint={2202.10873},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2202.10873}, 
}
```

## License

Released under the terms of [License.txt](License.txt).
