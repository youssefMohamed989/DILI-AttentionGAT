"""
17_repeated_cv_chem_informed.py
Section 3.13 / Table 11 / Figure 15. Repeats the scaffold-split
regeneration and GAT-CI-MT training procedure across the same five
independent splits used elsewhere, recording per-split test AUROC and the
pooled attention-enrichment ratio (across all seven alerts), to confirm
the chemistry-informed attention-alignment gain is not an artifact of the
primary split.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import (
    RESULTS_DIR, FIGURES_DIR, REPEATED_SPLIT_SEEDS, REPEATED_CV_PATIENCE,
    REPEATED_CV_MAX_EPOCHS, CHEM_INFORMED_LAMBDA, set_seed,
)
from dataset_io import dataframe_to_graphs, toxicophore_labels
from gnn_model import collate_graphs
from gnn_model_v2 import train_single_model, predict_proba
from gnn_model_v3 import DILI_GAT_CI_MT
from repeated_split_utils import full_curated_dataframe, make_split

import importlib
tox_module = importlib.import_module("14_toxicophore_attention_validation")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_df = full_curated_dataframe()

    per_split_results = []
    for split_seed in REPEATED_SPLIT_SEEDS:
        print(f"\n=== chem-informed repeated split, seed={split_seed} ===")
        set_seed(split_seed)
        train_df, val_df, test_df, _ = make_split(full_df, seed=split_seed)

        train_graphs, tr_mask = dataframe_to_graphs(train_df)
        val_graphs, va_mask = dataframe_to_graphs(val_df)
        test_graphs, te_mask = dataframe_to_graphs(test_df)
        train_df = train_df.loc[tr_mask].reset_index(drop=True)
        val_df = val_df.loc[va_mask].reset_index(drop=True)
        test_df = test_df.loc[te_mask].reset_index(drop=True)

        train_labels = train_df["DILI_label"].values.astype(np.float32)
        val_labels = val_df["DILI_label"].values.astype(np.float32)
        test_labels = test_df["DILI_label"].values.astype(np.float32)
        train_aux = toxicophore_labels(train_df)
        val_aux = toxicophore_labels(val_df)

        model = DILI_GAT_CI_MT()
        model, _, _ = train_single_model(
            model, train_graphs, train_labels, val_graphs, val_labels,
            collate_fn=collate_graphs, device=device,
            max_epochs=REPEATED_CV_MAX_EPOCHS, patience=REPEATED_CV_PATIENCE,
            aux_train_labels=train_aux, aux_val_labels=val_aux, aux_lambda=CHEM_INFORMED_LAMBDA,
            verbose=False,
        )
        test_prob = predict_proba(model, test_graphs, collate_graphs, device=device)
        test_auroc = roc_auc_score(test_labels, test_prob)

        summary, _, _, _ = tox_module.run_validation(model, full_df, device=device)
        pooled_enrichment = summary["pooled_enrichment_ratio"]

        print(f"  test AUROC={test_auroc:.4f} | pooled attention-enrichment={pooled_enrichment:.3f}")
        per_split_results.append({
            "split_seed": split_seed,
            "test_auroc": float(test_auroc),
            "pooled_enrichment_ratio": float(pooled_enrichment),
        })

    aurocs = np.array([r["test_auroc"] for r in per_split_results])
    enrichments = np.array([r["pooled_enrichment_ratio"] for r in per_split_results])
    summary_out = {
        "per_split": per_split_results,
        "mean_test_auroc": float(aurocs.mean()), "std_test_auroc": float(aurocs.std()),
        "mean_pooled_enrichment": float(enrichments.mean()), "std_pooled_enrichment": float(enrichments.std()),
    }
    with open(os.path.join(RESULTS_DIR, "table11_repeated_cv_chem_informed.json"), "w") as f:
        json.dump(summary_out, f, indent=2)
    print(json.dumps(summary_out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(per_split_results))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(x, aurocs, color="tab:blue", alpha=0.7, label="test AUROC")
    ax1.set_ylabel("Test AUROC", color="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"split {i+1}" for i in x])
    ax2 = ax1.twinx()
    ax2.plot(x, enrichments, color="tab:red", marker="o", label="pooled enrichment ratio")
    ax2.set_ylabel("Pooled attention-enrichment ratio", color="tab:red")
    plt.title("Figure 15. GAT-CI-MT across 5 independent scaffold splits")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure15_repeated_cv_chem_informed.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
