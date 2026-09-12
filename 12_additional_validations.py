"""
12_additional_validations.py
Additional robustness checks referenced in the Results/Discussion:
  - Y-randomization (label permutation) test: retrains the single-model
    GAT with shuffled training labels; a properly-behaved model should
    perform near chance on the untouched test set.
  - Learning curve: test AUROC as a function of training-set fraction.
  - Applicability-domain analysis: nearest-neighbor Tanimoto similarity
    (Morgan fingerprints) of each test compound to the training set,
    to flag predictions made outside the model's chemical domain.
  - McNemar's test: paired comparison of the GAT's and RF baseline's
    per-molecule correctness on the test set.
"""
import json
import os

import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem import AllChem, Chem
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from common import RESULTS_DIR, SEED, set_seed, MAX_EPOCHS, EARLY_STOP_PATIENCE
from dataset_io import load_graphs_and_labels
from gnn_model import DILI_GAT, collate_graphs
from gnn_model_v2 import train_single_model, predict_proba


def y_randomization_test(train_graphs, train_labels, val_graphs, val_labels,
                          test_graphs, test_labels, device, n_repeats=3):
    aurocs = []
    rng = np.random.RandomState(SEED)
    for i in range(n_repeats):
        shuffled = rng.permutation(train_labels)
        model = DILI_GAT()
        model, _, _ = train_single_model(
            model, train_graphs, shuffled, val_graphs, val_labels,
            collate_fn=collate_graphs, device=device,
            max_epochs=min(MAX_EPOCHS, 100), patience=min(EARLY_STOP_PATIENCE, 15), verbose=False,
        )
        prob = predict_proba(model, test_graphs, collate_graphs, device=device)
        aurocs.append(roc_auc_score(test_labels, prob))
    return aurocs


def learning_curve(train_graphs, train_labels, val_graphs, val_labels,
                    test_graphs, test_labels, device, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    rng = np.random.RandomState(SEED)
    n = len(train_graphs)
    order = rng.permutation(n)
    curve = []
    for frac in fractions:
        k = max(int(round(frac * n)), 10)
        idx = order[:k]
        sub_graphs = [train_graphs[i] for i in idx]
        sub_labels = train_labels[idx]
        model = DILI_GAT()
        model, _, _ = train_single_model(
            model, sub_graphs, sub_labels, val_graphs, val_labels,
            collate_fn=collate_graphs, device=device,
            max_epochs=min(MAX_EPOCHS, 100), patience=min(EARLY_STOP_PATIENCE, 15), verbose=False,
        )
        prob = predict_proba(model, test_graphs, collate_graphs, device=device)
        curve.append({"train_fraction": frac, "n_train": k, "test_auroc": float(roc_auc_score(test_labels, prob))})
    return curve


def applicability_domain(train_df, test_df, smiles_col="canonical_smiles"):
    def fp(smi):
        mol = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    train_fps = [fp(s) for s in train_df[smiles_col]]
    max_sims = []
    for s in test_df[smiles_col]:
        test_fp = fp(s)
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_sims.append(max(sims))
    return np.array(max_sims)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    train_graphs, train_labels, train_df = load_graphs_and_labels("train")
    val_graphs, val_labels, _ = load_graphs_and_labels("val")
    test_graphs, test_labels, test_df = load_graphs_and_labels("test")

    print("Running Y-randomization test...")
    y_rand_aurocs = y_randomization_test(train_graphs, train_labels, val_graphs, val_labels,
                                          test_graphs, test_labels, device)
    print(f"  Y-randomization test AUROC (should be ~0.5): {y_rand_aurocs}")

    print("Computing learning curve...")
    lc = learning_curve(train_graphs, train_labels, val_graphs, val_labels,
                         test_graphs, test_labels, device)

    print("Computing applicability-domain (Tanimoto) analysis...")
    max_sims = applicability_domain(train_df, test_df)

    print("Running McNemar's test (GAT vs RF)...")
    gat_prob = np.load(os.path.join(RESULTS_DIR, "gat_single_test_probs.npy"))
    rf_prob = np.load(os.path.join(RESULTS_DIR, "rf_morgan_test_probs.npy"))
    gat_correct = ((gat_prob >= 0.5).astype(int) == test_labels.astype(int))
    rf_correct = ((rf_prob >= 0.5).astype(int) == test_labels.astype(int))
    both_correct = int((gat_correct & rf_correct).sum())
    gat_only = int((gat_correct & ~rf_correct).sum())
    rf_only = int((~gat_correct & rf_correct).sum())
    neither = int((~gat_correct & ~rf_correct).sum())
    table = [[both_correct, gat_only], [rf_only, neither]]
    mcnemar_result = mcnemar(table, exact=True)

    out = {
        "y_randomization_test_auroc": y_rand_aurocs,
        "learning_curve": lc,
        "applicability_domain": {
            "mean_max_tanimoto_to_train": float(max_sims.mean()),
            "frac_test_below_0.4_similarity": float((max_sims < 0.4).mean()),
        },
        "mcnemar_test": {
            "contingency_table": table,
            "statistic": float(mcnemar_result.statistic),
            "p_value": float(mcnemar_result.pvalue),
        },
    }
    with open(os.path.join(RESULTS_DIR, "additional_validations.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
