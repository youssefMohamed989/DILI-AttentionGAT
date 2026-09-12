"""
dataset_io.py
Small shared loading helper used by every training/analysis script to turn
a processed partition CSV (train.csv/val.csv/test.csv, produced by
01_data_prep.py) into a list of molecular graphs + label array. This is
infrastructure glue, not one of the numbered analysis stages.
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd

from common import PROCESSED_DATA_DIR, TOXICOPHORE_SMARTS, TOXICOPHORE_NAMES
from graph_features import mol_to_graph
from rdkit import Chem


def load_partition(name: str, processed_dir: str = PROCESSED_DATA_DIR) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"{name}.csv")
    return pd.read_csv(path)


def dataframe_to_graphs(df: pd.DataFrame, smiles_col: str = "canonical_smiles"):
    graphs, keep_mask = [], []
    for smi in df[smiles_col]:
        g = mol_to_graph(smi)
        graphs.append(g)
        keep_mask.append(g is not None)
    keep_mask = np.array(keep_mask)
    graphs = [g for g in graphs if g is not None]
    return graphs, keep_mask


def load_graphs_and_labels(name: str, processed_dir: str = PROCESSED_DATA_DIR
                            ) -> Tuple[list, np.ndarray, pd.DataFrame]:
    df = load_partition(name, processed_dir)
    graphs, keep_mask = dataframe_to_graphs(df)
    df = df.loc[keep_mask].reset_index(drop=True)
    labels = df["DILI_label"].values.astype(np.float32)
    return graphs, labels, df


_TOXICOPHORE_PATTERNS = {name: Chem.MolFromSmarts(smarts)
                          for name, smarts in TOXICOPHORE_SMARTS.items()}


def toxicophore_labels(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> np.ndarray:
    """Returns (n_mols, 7) binary matrix flagging which of the seven
    structural alerts each molecule matches (molecule-level presence, used
    as the auxiliary multi-task target and for alert-count correlations)."""
    out = np.zeros((len(df), len(TOXICOPHORE_NAMES)), dtype=np.float32)
    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, name in enumerate(TOXICOPHORE_NAMES):
            patt = _TOXICOPHORE_PATTERNS[name]
            if patt is not None and mol.HasSubstructMatch(patt):
                out[i, j] = 1.0
    return out


def toxicophore_atom_flags(mol) -> np.ndarray:
    """Per-atom, per-alert binary flags for a single RDKit Mol: (n_atoms, 7)."""
    n_atoms = mol.GetNumAtoms()
    flags = np.zeros((n_atoms, len(TOXICOPHORE_NAMES)), dtype=bool)
    for j, name in enumerate(TOXICOPHORE_NAMES):
        patt = _TOXICOPHORE_PATTERNS[name]
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            for atom_idx in match:
                flags[atom_idx, j] = True
    return flags
