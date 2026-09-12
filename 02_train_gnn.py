"""
02_train_gnn.py
Trains the single-model, 3-layer GAT (gnn_model.DILI_GAT) on the primary
scaffold split (Section 2.6): Adam (lr 5e-4, weight decay 1e-5),
class-weighted BCE, batch size 32, gradient-norm clipping at 5.0, early
stopping on validation AUROC (patience 25), checkpointing the best model.

This reproduces the "GAT, single model (this work)" row of Table 2.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import CHECKPOINT_DIR, RESULTS_DIR, EARLY_STOP_PATIENCE, MAX_EPOCHS, BATCH_SIZE, set_seed, SEED
from dataset_io import load_graphs_and_labels
from gnn_model import DILI_GAT, collate_graphs
from gnn_model_v2 import compute_pos_weight, train_single_model, predict_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    set_seed(args.seed)

    train_graphs, train_labels, _ = load_graphs_and_labels("train")
    val_graphs, val_labels, _ = load_graphs_and_labels("val")
    test_graphs, test_labels, test_df = load_graphs_and_labels("test")

    print(f"train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")

    model = DILI_GAT()
    model, best_val_auroc, history = train_single_model(
        model, train_graphs, train_labels, val_graphs, val_labels,
        collate_fn=collate_graphs, device=args.device,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=BATCH_SIZE,
    )

    test_prob = predict_proba(model, test_graphs, collate_graphs, device=args.device)
    test_auroc = roc_auc_score(test_labels, test_prob)
    print(f"Best val AUROC: {best_val_auroc:.4f} | Test AUROC (single GAT): {test_auroc:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "gat_single_model.pt")
    torch.save(model.state_dict(), ckpt_path)

    np.save(os.path.join(RESULTS_DIR, "gat_single_test_probs.npy"), test_prob)
    with open(os.path.join(RESULTS_DIR, "gat_single_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "gat_single_summary.json"), "w") as f:
        json.dump({
            "best_val_auroc": float(best_val_auroc),
            "test_auroc": float(test_auroc),
            "n_train": len(train_graphs), "n_val": len(val_graphs), "n_test": len(test_graphs),
            "checkpoint": ckpt_path,
        }, f, indent=2)


if __name__ == "__main__":
    main()
