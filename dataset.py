"""Dataset helpers for Ligandformer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

from .featurization import AtomVocabulary, smiles_to_graph


SMILES_COLUMNS = ("cleaned_smiles", "Cleaned_SMILES", "smiles", "SMILES")
LABEL_COLUMNS = ("label", "Label", "y", "Y")


def find_column(columns: Iterable[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    column_set = {str(column): column for column in columns}
    lower_map = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {', '.join(candidates)}")
    return None


def read_molecule_csv(path: str | Path) -> Tuple[pd.DataFrame, str, Optional[str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    smiles_col = find_column(df.columns, SMILES_COLUMNS, required=True)
    label_col = find_column(df.columns, LABEL_COLUMNS, required=False)
    df = df[df[smiles_col].notna()].copy()
    if label_col is not None:
        df = df[df[label_col].notna()].copy()
    return df, smiles_col, label_col


def collect_smiles(paths: Sequence[str | Path]) -> List[str]:
    smiles = []
    for path in paths:
        df, smiles_col, _ = read_molecule_csv(path)
        smiles.extend(df[smiles_col].astype(str).tolist())
    return smiles


class MoleculeGraphDataset:
    """In-memory list-style PyG dataset built directly from a CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        atom_vocab: AtomVocabulary,
        add_hydrogens: bool = True,
        require_labels: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.atom_vocab = atom_vocab
        self.add_hydrogens = add_hydrogens
        self.frame, self.smiles_col, self.label_col = read_molecule_csv(self.csv_path)
        if require_labels and self.label_col is None:
            raise ValueError(f"{self.csv_path} does not contain a label column.")
        self.graphs = self._build_graphs(require_labels=require_labels)

    def _build_graphs(self, require_labels: bool) -> List[Data]:
        graphs = []
        for row_id, row in self.frame.reset_index(drop=True).iterrows():
            smiles = str(row[self.smiles_col])
            try:
                x, edge_index, edge_attr = smiles_to_graph(
                    smiles,
                    atom_vocab=self.atom_vocab,
                    add_hydrogens=self.add_hydrogens,
                )
            except ValueError as exc:
                raise ValueError(f"Failed to featurize row {row_id} in {self.csv_path}: {exc}") from exc

            label = None
            if self.label_col is not None:
                label = torch.tensor([float(row[self.label_col])], dtype=torch.float32)
            elif require_labels:
                raise ValueError(f"Missing label at row {row_id} in {self.csv_path}.")

            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=label,
                smiles=smiles,
                row_id=row_id,
            )
            graphs.append(graph)
        return graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
