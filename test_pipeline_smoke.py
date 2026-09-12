"""
tests/test_pipeline_smoke.py
Lightweight smoke tests that don't require the real gold-standard DILI
dataset: they check that featurization, batching, and a forward/backward
pass work for each of the three model variants, and that the toxicophore
SMARTS patterns all parse and match at least one test molecule.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from rdkit import Chem

from common import TOXICOPHORE_SMARTS
from graph_features import mol_to_graph, morgan_fingerprint, rdkit_descriptors
from gnn_model import DILI_GAT, collate_graphs
from gnn_model_v2 import DILI_GAT_JK
from gnn_model_v3 import DILI_GAT_CI_MT

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "Nc1ccccc1",                # aniline
    "NNC(=O)c1ccccc1",          # hydrazide
    "c1ccsc1",                  # thiophene
    "Clc1ccccc1",               # chlorobenzene
]


def _graphs():
    return [mol_to_graph(s) for s in SMILES]


def test_mol_to_graph_shapes():
    for g in _graphs():
        assert g is not None
        assert g["node_feats"].shape[1] == 37
        assert g["edge_feats"].shape[1] == 6
        assert g["edge_index"].shape[0] == 2


def test_morgan_and_descriptors():
    fp = morgan_fingerprint(SMILES[0])
    assert fp.shape == (2048,)
    desc, names = rdkit_descriptors(SMILES[0])
    assert desc.shape[0] == len(names)
    assert desc.shape[0] > 100


def test_toxicophore_smarts_parse_and_match():
    any_match = False
    for name, smarts in TOXICOPHORE_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        assert patt is not None, f"SMARTS failed to parse: {name}"
        for smi in SMILES:
            mol = Chem.MolFromSmiles(smi)
            if mol.HasSubstructMatch(patt):
                any_match = True
    assert any_match


def _batch_and_labels():
    graphs = _graphs()
    labels = np.array([0, 1, 1, 0, 1], dtype=np.float32)
    batch = collate_graphs(graphs, labels=labels)
    return batch


def test_forward_backward_gat():
    batch = _batch_and_labels()
    model = DILI_GAT()
    logit = model(batch)
    assert logit.shape == (5,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, batch.labels)
    loss.backward()


def test_forward_backward_gat_jk():
    batch = _batch_and_labels()
    model = DILI_GAT_JK()
    logit, alpha = model(batch, return_attention=True)
    assert logit.shape == (5,)
    assert alpha.shape[0] == batch.node_feats.shape[0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, batch.labels)
    loss.backward()


def test_forward_backward_gat_ci_mt():
    batch = _batch_and_labels()
    aux_labels = np.random.RandomState(0).randint(0, 2, size=(5, 7)).astype(np.float32)
    batch.aux_labels = torch.as_tensor(aux_labels)
    model = DILI_GAT_CI_MT()
    dili_logit, alpha, aux_logits = model.forward_multitask(batch)
    assert dili_logit.shape == (5,)
    assert aux_logits.shape == (5, 7)
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(dili_logit, batch.labels)
        + 0.4 * torch.nn.functional.binary_cross_entropy_with_logits(aux_logits, batch.aux_labels)
    )
    loss.backward()
