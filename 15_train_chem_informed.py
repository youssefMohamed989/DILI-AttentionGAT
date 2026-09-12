"""
15_train_chem_informed.py
Section 2.13 / Table 8 / Figure 14. Trains the chemistry-informed
multi-task GAT (GAT-CI-MT, gnn_model_v3.DILI_GAT_CI_MT) on the primary
scaffold split with the combined loss L = L_DILI + lambda * L_toxicophore
(lambda = 0.4), then repeats the toxicophore-attention alignment test of
14_toxicophore_attention_validation.py on its attention weights, for
direct comparison to the single-task baseline GAT.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import (
    RESULTS_DIR, CHECKPOINT_DIR, FIGURES_DIR, CHEM_INFORMED_LAMBDA, set_seed, SEED,
)
from dataset_io import load_partition, dataframe_to_graphs, toxicophore_labels
from gnn_model import collate_graphs
from gnn_model_v2 import train_single_model, predict_proba
from gnn_model_v3 import DILI_GAT_CI_MT

import importlib
tox_module = importlib.import_module("14_toxicophore_attention_validation")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    train_df = load_partition("train")
    val_df = load_partition("val")
    test_df = load_partition("test")

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
    model, best_val_auroc, history = train_single_model(
        model, train_graphs, train_labels, val_graphs, val_labels,
        collate_fn=collate_graphs, device=device,
        aux_train_labels=train_aux, aux_val_labels=val_aux,
        aux_lambda=CHEM_INFORMED_LAMBDA,
    )

    test_prob = predict_proba(model, test_graphs, collate_graphs, device=device)
    test_auroc = roc_auc_score(test_labels, test_prob)
    print(f"GAT-CI-MT | best val AUROC={best_val_auroc:.4f} | test AUROC={test_auroc:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "gat_ci_mt.pt")
    torch.save(model.state_dict(), ckpt_path)
    np.save(os.path.join(RESULTS_DIR, "gat_ci_mt_test_probs.npy"), test_prob)
    with open(os.path.join(RESULTS_DIR, "gat_ci_mt_summary.json"), "w") as f:
        json.dump({"best_val_auroc": float(best_val_auroc), "test_auroc": float(test_auroc)}, f, indent=2)

    # repeat the toxicophore-attention alignment test (Table 8) on the full curated dataset
    from repeated_split_utils import full_curated_dataframe
    full_df = full_curated_dataframe()
    summary, prob, alert_labels, dili_labels = tox_module.run_validation(model, full_df, device=device)
    with open(os.path.join(RESULTS_DIR, "table8_toxicophore_validation_chem_informed.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Figure 14: side-by-side comparison with the baseline GAT, if available
    baseline_path = os.path.join(RESULTS_DIR, "table7_toxicophore_validation_baseline_gat.json")
    if os.path.exists(baseline_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with open(baseline_path) as f:
            baseline_summary = json.load(f)

        names = list(summary["table7_per_alert"].keys())
        base_enrich = [baseline_summary["table7_per_alert"].get(n, {}).get("enrichment_ratio", np.nan) for n in names]
        chem_enrich = [summary["table7_per_alert"][n]["enrichment_ratio"] for n in names]
        labels_disp = [summary["table7_per_alert"][n]["display_name"] for n in names]

        x = np.arange(len(names))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - width / 2, base_enrich, width, label="Baseline GAT")
        plt.bar(x + width / 2, chem_enrich, width, label="GAT-CI-MT")
        plt.xticks(x, labels_disp, rotation=30, ha="right", fontsize=8)
        plt.axhline(1.0, color="gray", linestyle="--")
        plt.ylabel("Attention enrichment ratio")
        plt.title("Figure 14. Baseline vs. chemistry-informed attention enrichment")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure14_baseline_vs_chem_informed.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
