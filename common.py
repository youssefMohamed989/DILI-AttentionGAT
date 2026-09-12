"""
common.py
Shared constants, paths, and utility functions used by every stage of the
DILI-GAT pipeline (Validating and Correcting Graph Neural Network Attention
for Drug-Induced Liver Injury Prediction).

Nothing in this file trains a model or reads chemistry files itself; it just
centralizes the numbers/paths so every script agrees on them.
"""
import os
import random
import json

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED):
    """Seed python, numpy, and torch (if importable) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
    os.makedirs(_d, exist_ok=True)

RAW_DATASET_PATH = os.path.join(RAW_DATA_DIR, "gold_standard_dili.csv")

# --------------------------------------------------------------------------
# Dataset curation constants (Section 2.1)
# --------------------------------------------------------------------------
MW_MIN = 100.0
MW_MAX = 1200.0
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train, val, test

# --------------------------------------------------------------------------
# Atom / bond featurization constants (Section 2.2)
# --------------------------------------------------------------------------
ATOM_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
# one-hot over ATOM_ELEMENTS + 1 "other" slot = 13
# + degree(1) + formal charge(1) + hybridization one-hot(5) + aromatic(1)
# + in-ring(1) + implicit/explicit H count(1) + total connected H(1)
# + degree bucket paddings -> total dimension is fixed at 37 (see graph_features.py)
ATOM_FEATURE_DIM = 37
BOND_FEATURE_DIM = 6

# --------------------------------------------------------------------------
# GAT architecture constants (Section 2.3)
# --------------------------------------------------------------------------
GAT_NUM_LAYERS = 3
GAT_NUM_HEADS = 4
GAT_HIDDEN_DIM = 64
GAT_DROPOUT = 0.2

# --------------------------------------------------------------------------
# Training hyperparameters (Section 2.6)
# --------------------------------------------------------------------------
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
GRAD_CLIP_NORM = 5.0
EARLY_STOP_PATIENCE = 25
MAX_EPOCHS = 300

# Repeated scaffold-split CV (Section 2.7) uses a shorter schedule
REPEATED_CV_PATIENCE = 20
REPEATED_CV_MAX_EPOCHS = 100
N_REPEATED_SPLITS = 5
REPEATED_SPLIT_SEEDS = [42, 43, 44, 45, 46]

# Deep ensemble (Section 2.4)
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

# Chemistry-informed multi-task loss weight (Section 2.13)
CHEM_INFORMED_LAMBDA = 0.4

# Bootstrap (Section 2.8)
N_BOOTSTRAP = 5000

# --------------------------------------------------------------------------
# Seven literature-derived hepatotoxicity structural alerts ("toxicophores")
# (Section 2.12). SMARTS patterns are standard reactive-metabolite /
# structural-alert definitions consistent with Kalgutkar & Dalvie (2015) and
# Stepan et al. (2011); see Supplementary Table S1 in the manuscript.
# --------------------------------------------------------------------------
TOXICOPHORE_SMARTS = {
    "aniline_arylamine": "[NX3;H2,H1;!$(NC=O)][c]",
    "hydrazine_hydrazide": "[NX3][NX3]",
    "thiophene": "c1ccsc1",
    "michael_acceptor_enone": "[CX3]=[CX3][CX3]=[OX1]",
    "furan": "c1ccoc1",
    "aromatic_halide": "[c][F,Cl,Br,I]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
}
TOXICOPHORE_NAMES = list(TOXICOPHORE_SMARTS.keys())
N_TOXICOPHORES = len(TOXICOPHORE_NAMES)

DISPLAY_NAMES = {
    "aniline_arylamine": "Aniline / arylamine",
    "hydrazine_hydrazide": "Hydrazine/hydrazide",
    "thiophene": "Thiophene",
    "michael_acceptor_enone": "Michael acceptor",
    "furan": "Furan",
    "aromatic_halide": "Aromatic halide",
    "carboxylic_acid": "Carboxylic acid",
}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)
