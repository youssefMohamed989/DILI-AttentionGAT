"""
06_train_ensemble_gnn.py
Section 2.4 / Table 2 ("GAT, 5-model deep ensemble"). Trains a 5-member
deep ensemble of DILI_GAT_JK on the primary scaffold split, varying only
the seed (42-46) governing weight init and mini-batch shuffling. Final
predictions are the mean of the five members' probabilities; the
cross-member standard deviation is retained as a per-molecule
ensemble-disagreement score (used in Section 3.6 / 09_advanced_figures.py).
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import CHECKPOINT_DIR, RESULTS_DIR, ENSEMBLE_SEEDS, set_seed
from dataset_io import load_graphs_and_labels
from gnn_model import collate_graphs
from gnn_model_v2 import DILI_GAT_JK, train_single_model, DeepEnsemble


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_graphs, train_labels, _ = load_graphs_and_labels("train")
    val_graphs, val_labels, _ = load_graphs_and_labels("val")
    test_graphs, test_labels, _ = load_graphs_and_labels("test")

    members = []
    per_member_val_auroc = []
    for seed in ENSEMBLE_SEEDS:
        print(f"\n=== training ensemble member, seed={seed} ===")
        set_seed(seed)
        model = DILI_GAT_JK()
        model, best_val_auroc, _ = train_single_model(
            model, train_graphs, train_labels, val_graphs, val_labels,
            collate_fn=collate_graphs, device=device,
        )
        members.append(model)
        per_member_val_auroc.append(float(best_val_auroc))
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"gat_jk_ensemble_seed{seed}.pt"))

    ensemble = DeepEnsemble(members)
    mean_prob, std_prob, all_probs = ensemble.predict(test_graphs, collate_graphs, device=device)
    ensemble_auroc = roc_auc_score(test_labels, mean_prob)
    print(f"\nEnsemble test AUROC: {ensemble_auroc:.4f}")

    np.save(os.path.join(RESULTS_DIR, "ensemble_test_probs.npy"), mean_prob)
    np.save(os.path.join(RESULTS_DIR, "ensemble_test_probs_std.npy"), std_prob)
    np.save(os.path.join(RESULTS_DIR, "ensemble_test_probs_per_member.npy"), all_probs)

    with open(os.path.join(RESULTS_DIR, "ensemble_summary.json"), "w") as f:
        json.dump({
            "seeds": ENSEMBLE_SEEDS,
            "per_member_val_auroc": per_member_val_auroc,
            "ensemble_test_auroc": float(ensemble_auroc),
        }, f, indent=2)


if __name__ == "__main__":
    main()
