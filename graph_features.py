"""
graph_features.py
Atom/bond feature extraction for graph construction (Section 2.2), and
Morgan fingerprint / RDKit descriptor computation for the descriptor-based
baseline models.
"""
from typing import List, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from common import ATOM_ELEMENTS, ATOM_FEATURE_DIM, BOND_FEATURE_DIM

HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]


def _one_hot(value, choices) -> List[int]:
    return [int(value == c) for c in choices]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """37-dimensional atom feature vector.

    Layout:
      [0:12]  one-hot element in ATOM_ELEMENTS
      [12]    'other' element flag
      [13]    degree (heavy-atom neighbor count), raw int
      [14]    formal charge
      [15:20] one-hot hybridization (SP, SP2, SP3, SP3D, SP3D2)
      [20]    aromaticity flag
      [21]    ring-membership flag
      [22]    total (implicit+explicit) H count
      [23:37] reserved capacity: degree one-hot (0..5) + explicit valence
              one-hot (0..6) + chirality flag, zero-padded to a fixed
              37-dim vector for stable model input size.
    """
    symbol = atom.GetSymbol()
    element_oh = _one_hot(symbol, ATOM_ELEMENTS)
    other_flag = int(symbol not in ATOM_ELEMENTS)

    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybridization_oh = _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    aromatic = int(atom.GetIsAromatic())
    in_ring = int(atom.IsInRing())
    total_h = atom.GetTotalNumHs()

    degree_oh = _one_hot(min(degree, 5), list(range(6)))
    valence_oh = _one_hot(min(atom.GetValence(Chem.ValenceType.EXPLICIT), 6), list(range(7)))
    chiral_flag = int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)

    feats = (
        element_oh
        + [other_flag]
        + [degree]
        + [formal_charge]
        + hybridization_oh
        + [aromatic]
        + [in_ring]
        + [total_h]
        + degree_oh
        + valence_oh
        + [chiral_flag]
    )
    feats = np.array(feats, dtype=np.float32)
    if feats.shape[0] < ATOM_FEATURE_DIM:
        feats = np.pad(feats, (0, ATOM_FEATURE_DIM - feats.shape[0]))
    return feats[:ATOM_FEATURE_DIM]


BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """6-dimensional bond feature vector: one-hot bond order (4),
    conjugation flag, ring-membership flag."""
    bt_oh = _one_hot(bond.GetBondType(), BOND_TYPES)
    conj = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())
    feats = np.array(bt_oh + [conj, in_ring], dtype=np.float32)
    assert feats.shape[0] == BOND_FEATURE_DIM
    return feats


def mol_to_graph(smiles: str):
    """Convert a SMILES string to an attributed molecular graph.

    Returns a dict with:
      node_feats: (n_atoms, 37) float32
      edge_index: (2, n_edges) int64, explicit bidirectional edges
      edge_feats: (n_edges, 6) float32
      mol: the parsed RDKit Mol (with explicit Hs removed, atoms in
           canonical RDKit atom order)
    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = np.stack([atom_features(a) for a in mol.GetAtoms()], axis=0)

    src, dst, e_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # explicit bidirectional edges
        src += [i, j]
        dst += [j, i]
        e_feats += [bf, bf]

    if len(src) == 0:
        # single-atom molecule: add a self-loop so message passing is defined
        src, dst = [0], [0]
        e_feats = [np.zeros(BOND_FEATURE_DIM, dtype=np.float32)]

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_feats = np.stack(e_feats, axis=0)

    return {
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_feats": edge_feats,
        "mol": mol,
    }


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """2048-bit Morgan (ECFP-like) fingerprint, radius 2."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_descriptors(smiles: str) -> Tuple[np.ndarray, List[str]]:
    """Full set of RDKit 2D physicochemical/topological descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    names = [name for name, _ in Descriptors._descList]
    values = []
    for name, func in Descriptors._descList:
        try:
            values.append(func(mol))
        except Exception:
            values.append(np.nan)
    return np.array(values, dtype=np.float64), names


def molecular_weight(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def canonical_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def murcko_scaffold(smiles: str):
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold_mol, canonical=True)
