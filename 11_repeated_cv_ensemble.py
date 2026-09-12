"""
11_repeated_cv_ensemble.py
Section 3.7, definitive robustness check. Repeats the full 5-member
deep-ensemble training procedure (Section 2.4) across the same five
independent scaffold splits used in 07_repeated_scaffold_cv.py (25 total
GAT trainings in all), to test whether ensembling closes any single-model
generalization gap to the RF baseline observed at that level.
"""
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

from common import (
    RESULTS_DIR, REPEATED_SPLIT_SEEDS, ENSEMBLE_SEEDS, REPEATED_CV_PATIENCE,
    REPEATED_CV_MAX_EPOCHS, SEED, set_seed,
)
from dataset_io import dataframe_to_graphs
from graph_features import morgan_fingerprint
from gnn_model import collate_graphs
from gnn_model_v2 import DILI_GAT_JK, train_single_model, DeepEnsemble
from repeated_split_utils import full_curated_dataframe, make_split


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_df = full_curated_dataframe()

    ensemble_aurocs, rf_aurocs = [], []
    per_split_results = []

    for split_seed in REPEATED_SPLIT_SEEDS:
        print(f"\n=== repeated split (ensemble), split_seed={split_seed} ===")
        set_seed(split_seed)
        train_df, val_df, test_df, _ = make_split(full_df, seed=split_seed)

        train_graphs, tr_mask = dataframe_to_graphs(train_df)
        val_graphs, va_mask = dataframe_to_graphs(val_df)
        test_graphs, te_mask = dataframe_to_graphs(test_df)
        train_labels = train_df.loc[tr_mask, "DILI_label"].values.astype(np.float32)
        val_labels = val_df.loc[va_mask, "DILI_label"].values.astype(np.float32)
        test_labels = test_df.loc[te_mask, "DILI_label"].values.astype(np.float32)

        members = []
        for member_seed in ENSEMBLE_SEEDS:
            set_seed(member_seed)
            model = DILI_GAT_JK()
            model, _, _ = train_single_model(
                model, train_graphs, train_labels, val_graphs, val_labels,
                collate_fn=collate_graphs, device=device,
                max_epochs=REPEATED_CV_MAX_EPOCHS, patience=REPEATED_CV_PATIENCE,
                verbose=False,
            )
            members.append(model)

        ensemble = DeepEnsemble(members)
        mean_prob, _, _ = ensemble.predict(test_graphs, collate_graphs, device=device)
        ens_auroc = roc_auc_score(test_labels, mean_prob)
        ensemble_aurocs.append(ens_auroc)

        X_train = np.stack([morgan_fingerprint(s) for s in train_df.loc[tr_mask, "canonical_smiles"]])
        X_test = np.stack([morgan_fingerprint(s) for s in test_df.loc[te_mask, "canonical_smiles"]])
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        rf.fit(X_train, train_labels)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_auroc = roc_auc_score(test_labels, rf_prob)
        rf_aurocs.append(rf_auroc)

        print(f"  ensemble AUROC={ens_auroc:.4f} | RF AUROC={rf_auroc:.4f}")
        per_split_results.append({
            "split_seed": split_seed, "ensemble_test_auroc": float(ens_auroc),
            "rf_morgan_test_auroc": float(rf_auroc),
        })

    ensemble_aurocs, rf_aurocs = np.array(ensemble_aurocs), np.array(rf_aurocs)
    t_stat, p_value = stats.ttest_rel(ensemble_aurocs, rf_aurocs)

    summary = {
        "per_split": per_split_results,
        "ensemble_mean_auroc": float(ensemble_aurocs.mean()),
        "ensemble_std_auroc": float(ensemble_aurocs.std()),
        "rf_morgan_mean_auroc": float(rf_aurocs.mean()),
        "rf_morgan_std_auroc": float(rf_aurocs.std()),
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(p_value),
    }
    with open(os.path.join(RESULTS_DIR, "repeated_cv_ensemble.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
