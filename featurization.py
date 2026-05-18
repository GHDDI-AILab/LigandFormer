"""SMILES featurization utilities for Ligandformer.

The node featurization follows the seven atomic attributes described in the
Ligandformer paper (Figure 1c): atom symbol, degree, formal charge, chirality,
number of attached hydrogens, hybridization, and aromaticity. Each attribute is
encoded as a one-hot (or scalar) slice, concatenated into the initial node
feature vector ``f_init`` that is fed to the node embedding layer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import rdchem


DEFAULT_ATOMS = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]


DEGREE_BINS = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_BINS = [-2, -1, 0, 1, 2]
NUM_HS_BINS = [0, 1, 2, 3, 4]
CHIRALITY_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
]
HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.UNSPECIFIED,
]


def _extra_feature_dim() -> int:
    # degree + formal charge + chirality + num_hs + hybridization + aromatic
    return (
        (len(DEGREE_BINS) + 1)
        + (len(FORMAL_CHARGE_BINS) + 1)
        + len(CHIRALITY_TAGS)
        + (len(NUM_HS_BINS) + 1)
        + len(HYBRIDIZATION_TYPES)
        + 1
    )


def node_feature_dim(vocab_size: int) -> int:
    """Total dimensionality of the initial node feature vector f_init."""

    return vocab_size + _extra_feature_dim()


@dataclass
class AtomVocabulary:
    """A small serializable atom-symbol vocabulary."""

    token_to_idx: Dict[str, int]
    unk_token: str = "<unk>"

    @classmethod
    def from_smiles(
        cls,
        smiles_values: Iterable[str],
        add_hydrogens: bool = True,
        include_defaults: bool = True,
    ) -> "AtomVocabulary":
        tokens = set(DEFAULT_ATOMS if include_defaults else [])
        for smiles in smiles_values:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            if add_hydrogens:
                mol = Chem.AddHs(mol)
            tokens.update(atom.GetSymbol() for atom in mol.GetAtoms())
        ordered = sorted(tokens)
        token_to_idx = {token: idx for idx, token in enumerate(ordered)}
        token_to_idx[cls.unk_token] = len(token_to_idx)
        return cls(token_to_idx=token_to_idx)

    @classmethod
    def from_dict(cls, token_to_idx: Dict[str, int]) -> "AtomVocabulary":
        vocab = dict(token_to_idx)
        if cls.unk_token not in vocab:
            vocab[cls.unk_token] = len(vocab)
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def encode(self, symbol: str) -> int:
        if symbol not in self.token_to_idx:
            warnings.warn(f"{symbol} is not in atom vocabulary; using <unk>.", RuntimeWarning)
        return self.token_to_idx.get(symbol, self.token_to_idx[self.unk_token])

    def to_dict(self) -> Dict[str, int]:
        return dict(self.token_to_idx)


def canonicalize_molecule(mol: Chem.Mol, add_hydrogens: bool = True) -> Chem.Mol:
    if mol is None:
        raise ValueError("Cannot canonicalize an invalid molecule.")
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol))
    order = sorted(range(len(ranks)), key=lambda idx: ranks[idx])
    mol = Chem.RenumberAtoms(mol, order)
    mol.UpdatePropertyCache(strict=False)
    return mol


def smiles_to_graph(
    smiles: str,
    atom_vocab: AtomVocabulary,
    add_hydrogens: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a SMILES string into node features, edge indices, and edge features."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = canonicalize_molecule(mol, add_hydrogens=add_hydrogens)
    node_features = get_node_features(mol, atom_vocab)
    edge_features = get_edge_features(mol)
    edge_index = get_adjacency_info(mol)
    return node_features, edge_index, edge_features


def _one_hot(value, choices: List) -> List[float]:
    encoding = [0.0] * (len(choices) + 1)
    try:
        encoding[choices.index(value)] = 1.0
    except ValueError:
        encoding[-1] = 1.0
    return encoding


def _one_hot_strict(value, choices: List) -> List[float]:
    encoding = [0.0] * len(choices)
    try:
        encoding[choices.index(value)] = 1.0
    except ValueError:
        pass
    return encoding


def get_node_features(mol: Chem.Mol, atom_vocab: AtomVocabulary) -> torch.Tensor:
    """Return a (N, D) float tensor with the seven concatenated atom attributes."""

    vocab_size = len(atom_vocab)
    features: List[List[float]] = []
    for atom in mol.GetAtoms():
        symbol_onehot = [0.0] * vocab_size
        symbol_onehot[atom_vocab.encode(atom.GetSymbol())] = 1.0
        row: List[float] = []
        row.extend(symbol_onehot)
        row.extend(_one_hot(atom.GetDegree(), DEGREE_BINS))
        row.extend(_one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_BINS))
        row.extend(_one_hot_strict(atom.GetChiralTag(), CHIRALITY_TAGS))
        row.extend(_one_hot(atom.GetTotalNumHs(includeNeighbors=True), NUM_HS_BINS))
        row.extend(_one_hot_strict(atom.GetHybridization(), HYBRIDIZATION_TYPES))
        row.append(1.0 if atom.GetIsAromatic() else 0.0)
        features.append(row)
    if not features:
        return torch.empty((0, node_feature_dim(vocab_size)), dtype=torch.float)
    return torch.tensor(features, dtype=torch.float)


def get_edge_features(mol: Chem.Mol) -> torch.Tensor:
    edge_features: List[List[float]] = []
    for bond in mol.GetBonds():
        values = [float(bond.GetBondTypeAsDouble()), float(bond.IsInRing())]
        edge_features.extend([values, values])
    if not edge_features:
        return torch.empty((0, 2), dtype=torch.float)
    return torch.tensor(edge_features, dtype=torch.float)


def get_adjacency_info(mol: Chem.Mol) -> torch.Tensor:
    edge_indices: List[List[int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
    if not edge_indices:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()


def atom_attention_scores(attention_blocks: Optional[List[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Return integrated atom-level attention scores from block attention maps."""

    if not attention_blocks:
        return None
    scores = []
    for attention in attention_blocks:
        if attention.numel() == 0:
            continue
        scores.append(attention.mean(dim=1))
    if not scores:
        return None
    return torch.stack(scores, dim=0).mean(dim=0)
