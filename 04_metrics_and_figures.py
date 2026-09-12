"""
04_metrics_and_figures.py
Section 3.2 / Table 2 / Figure 1. Computes the primary metric suite
(AUROC, AUPRC, accuracy, sensitivity, specificity, MCC, F1) for the
single-model GAT and all three baselines on the primary scaffold-split
test set, and renders the ROC-curve figure and the GAT training-dynamics
figure.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, recall_score,
    matthews_corrcoef, f1_score, roc_curve, confusion_matrix,
)

from common import RESULTS_DIR, FIGURES_DIR
from dataset_io import load_graphs_and_labels


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) else float("nan")


def full_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred)),
        "Specificity": float(specificity_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
    }


def main():
    _, test_labels, _ = load_graphs_and_labels("test")

    model_probs = {
        "GAT, single model (this work)": np.load(os.path.join(RESULTS_DIR, "gat_single_test_probs.npy")),
        "RF + Morgan FP": np.load(os.path.join(RESULTS_DIR, "rf_morgan_test_probs.npy")),
        "LR + Morgan FP": np.load(os.path.join(RESULTS_DIR, "lr_morgan_test_probs.npy")),
        "RF + RDKit descriptors": np.load(os.path.join(RESULTS_DIR, "rf_descriptors_test_probs.npy")),
    }

    ensemble_path = os.path.join(RESULTS_DIR, "ensemble_test_probs.npy")
    if os.path.exists(ensemble_path):
        model_probs["GAT, 5-model deep ensemble (this work)"] = np.load(ensemble_path)

    table2 = {name: full_metrics(test_labels, probs) for name, probs in model_probs.items()}
    with open(os.path.join(RESULTS_DIR, "table2_test_metrics.json"), "w") as f:
        json.dump(table2, f, indent=2)
    print(json.dumps(table2, indent=2))

    # Figure 1: ROC curves
    plt.figure(figsize=(5, 5))
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(test_labels, probs)
        auc = table2[name]["AUROC"]
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Figure 1. ROC curves, primary scaffold-split test set (n=108)")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_roc_curves.png"), dpi=200)
    plt.close()

    # Training-dynamics figure, if history is available
    hist_path = os.path.join(RESULTS_DIR, "gat_single_train_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(hist["train_loss"], color="tab:blue", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(hist["val_auroc"], color="tab:red", label="val AUROC")
        ax2.set_ylabel("val AUROC", color="tab:red")
        plt.title("GAT training dynamics")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "training_dynamics.png"), dpi=200)
        plt.close()

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
