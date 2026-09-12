"""
07_repeated_scaffold_cv.py
Section 2.7 / Table 6 (single-model level). Repeats the scaffold-grouping
and 80:10:10 partitioning procedure five times with independent seeds,
retraining from scratch both a single-model GAT+JK and a random-forest /
Morgan-fingerprint baseline on each split, and records test-set AUROC.
Early-stopping patience is reduced to 20 epochs, max 100 epochs, for
computational tractability across the repeated experiment.
"""
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from common import (
    RESULTS_DIR, REPEATED_SPLIT_SEEDS, REPEATED_CV_PATIENCE, REPEATED_CV_MAX_EPOCHS, SEED, set_seed,
)
from dataset_io import dataframe_to_graphs
from graph_features import morgan_fingerprint
from gnn_model import collate_graphs
from gnn_model_v2 import DILI_GAT_JK, train_single_model, predict_proba
from repeated_split_utils import full_curated_dataframe, make_split


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_df = full_curated_dataframe()

    gat_aurocs, rf_aurocs = [], []
    per_split_results = []

    for split_seed in REPEATED_SPLIT_SEEDS:
        print(f"\n=== repeated split, seed={split_seed} ===")
        set_seed(split_seed)
        train_df, val_df, test_df, _ = make_split(full_df, seed=split_seed)

        train_graphs, tr_mask = dataframe_to_graphs(train_df)
        val_graphs, va_mask = dataframe_to_graphs(val_df)
        test_graphs, te_mask = dataframe_to_graphs(test_df)
        train_labels = train_df.loc[tr_mask, "DILI_label"].values.astype(np.float32)
        val_labels = val_df.loc[va_mask, "DILI_label"].values.astype(np.float32)
        test_labels = test_df.loc[te_mask, "DILI_label"].values.astype(np.float32)

        # GAT+JK
        model = DILI_GAT_JK()
        model, best_val_auroc, _ = train_single_model(
            model, train_graphs, train_labels, val_graphs, val_labels,
            collate_fn=collate_graphs, device=device,
            max_epochs=REPEATED_CV_MAX_EPOCHS, patience=REPEATED_CV_PATIENCE,
        )
        test_prob = predict_proba(model, test_graphs, collate_graphs, device=device)
        gat_auroc = roc_auc_score(test_labels, test_prob)
        gat_aurocs.append(gat_auroc)

        # RF + Morgan FP baseline, identical split
        X_train = np.stack([morgan_fingerprint(s) for s in train_df.loc[tr_mask, "canonical_smiles"]])
        X_test = np.stack([morgan_fingerprint(s) for s in test_df.loc[te_mask, "canonical_smiles"]])
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, train_labels)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_auroc = roc_auc_score(test_labels, rf_prob)
        rf_aurocs.append(rf_auroc)

        print(f"  GAT+JK test AUROC={gat_auroc:.4f} | RF+Morgan test AUROC={rf_auroc:.4f}")
        per_split_results.append({
            "split_seed": split_seed, "gat_jk_test_auroc": float(gat_auroc),
            "rf_morgan_test_auroc": float(rf_auroc), "n_test": int(len(test_labels)),
        })

    summary = {
        "per_split": per_split_results,
        "gat_jk_mean_auroc": float(np.mean(gat_aurocs)),
        "gat_jk_std_auroc": float(np.std(gat_aurocs)),
        "rf_morgan_mean_auroc": float(np.mean(rf_aurocs)),
        "rf_morgan_std_auroc": float(np.std(rf_aurocs)),
    }
    with open(os.path.join(RESULTS_DIR, "repeated_cv_single_model.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
