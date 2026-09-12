"""
16_mht_correction_and_aux_accuracy.py
Table 8 (corrected p-values) / Table 10. Applies Bonferroni and
Benjamini-Hochberg FDR correction for multiple testing across the seven
per-alert Mann-Whitney tests (both for the baseline GAT and for
GAT-CI-MT), and computes the auxiliary toxicophore-classification
accuracy (precision/recall/F1 per alert) of the trained chemistry-informed
model on the test partition.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from statsmodels.stats.multitest import multipletests

from common import RESULTS_DIR, CHECKPOINT_DIR, TOXICOPHORE_NAMES, DISPLAY_NAMES
from dataset_io import load_partition, dataframe_to_graphs, toxicophore_labels
from gnn_model import collate_graphs
from gnn_model_v3 import DILI_GAT_CI_MT


def correct_pvalues(table: dict):
    names = list(table.keys())
    pvals = [table[n]["mannwhitney_p_onesided"] for n in names]
    _, bonf, _, _ = multipletests(pvals, method="bonferroni")
    _, fdr, _, _ = multipletests(pvals, method="fdr_bh")
    out = {}
    for i, n in enumerate(names):
        out[n] = {
            **table[n],
            "p_bonferroni": float(bonf[i]),
            "p_fdr_bh": float(fdr[i]),
            "significant_after_correction": bool(bonf[i] < 0.05),
        }
    return out


def auxiliary_accuracy():
    test_df = load_partition("test")
    test_graphs, mask = dataframe_to_graphs(test_df)
    test_df = test_df.loc[mask].reset_index(drop=True)
    aux_labels = toxicophore_labels(test_df)

    model = DILI_GAT_CI_MT()
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gat_ci_mt.pt"), map_location="cpu"))
    model.eval()

    batch = collate_graphs(test_graphs)
    with torch.no_grad():
        _, _, aux_logits = model.forward_multitask(batch)
    aux_prob = torch.sigmoid(aux_logits).numpy()
    aux_pred = (aux_prob >= 0.5).astype(int)

    table10 = {}
    for j, name in enumerate(TOXICOPHORE_NAMES):
        y_true, y_pred = aux_labels[:, j].astype(int), aux_pred[:, j]
        n_positive = int(y_true.sum())
        table10[name] = {
            "display_name": DISPLAY_NAMES[name],
            "n_positive_in_test": n_positive,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    return table10


def main():
    for tag, path in [
        ("baseline", os.path.join(RESULTS_DIR, "table7_toxicophore_validation_baseline_gat.json")),
        ("chem_informed", os.path.join(RESULTS_DIR, "table8_toxicophore_validation_chem_informed.json")),
    ]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        corrected = correct_pvalues(summary["table7_per_alert"])
        summary["table7_per_alert_corrected"] = corrected
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{tag}] corrected p-values written back to {path}")

    if os.path.exists(os.path.join(CHECKPOINT_DIR, "gat_ci_mt.pt")):
        table10 = auxiliary_accuracy()
        with open(os.path.join(RESULTS_DIR, "table10_auxiliary_toxicophore_accuracy.json"), "w") as f:
            json.dump(table10, f, indent=2)
        print(json.dumps(table10, indent=2))


if __name__ == "__main__":
    main()
