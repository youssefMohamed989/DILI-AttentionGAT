"""
09_advanced_figures.py
Section 3.3-3.6. Figure 2 (bootstrap forest plot), precision-recall
curves, calibration curves, confusion matrices for all models, and the
ensemble-uncertainty-vs-correctness figure (Section 3.6).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

from common import RESULTS_DIR, FIGURES_DIR
from dataset_io import load_graphs_and_labels


def main():
    _, test_labels, _ = load_graphs_and_labels("test")

    model_files = {
        "GAT, single model": "gat_single_test_probs.npy",
        "GAT, 5-model ensemble": "ensemble_test_probs.npy",
        "RF + Morgan FP": "rf_morgan_test_probs.npy",
        "LR + Morgan FP": "lr_morgan_test_probs.npy",
        "RF + RDKit descriptors": "rf_descriptors_test_probs.npy",
    }
    probs = {n: np.load(os.path.join(RESULTS_DIR, f)) for n, f in model_files.items()
              if os.path.exists(os.path.join(RESULTS_DIR, f))}

    # --- Figure 2: bootstrap forest plot ---
    boot_path = os.path.join(RESULTS_DIR, "table3_bootstrap.json")
    if os.path.exists(boot_path):
        with open(boot_path) as f:
            table3 = json.load(f)["table3_bootstrap_ci"]
        names = list(table3.keys())
        points = [table3[n]["auroc_point_estimate"] for n in names]
        los = [table3[n]["auroc_point_estimate"] - table3[n]["ci_95_low"] for n in names]
        his = [table3[n]["ci_95_high"] - table3[n]["auroc_point_estimate"] for n in names]
        plt.figure(figsize=(6, 4))
        y = np.arange(len(names))
        plt.errorbar(points, y, xerr=[los, his], fmt="o", capsize=4)
        plt.yticks(y, names)
        plt.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Test AUROC (95% bootstrap CI)")
        plt.title("Figure 2. Bootstrap confidence intervals, primary split")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure2_bootstrap_forest.png"), dpi=200)
        plt.close()

    # --- Precision-recall curves ---
    plt.figure(figsize=(5, 5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(test_labels, p)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves, test set")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=200)
    plt.close()

    # --- Calibration curves ---
    plt.figure(figsize=(5, 5))
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(test_labels, p, n_bins=8, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curves, test set")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_curves.png"), dpi=200)
    plt.close()

    # --- Confusion matrices ---
    n_models = len(probs)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, p) in zip(axes, probs.items()):
        pred = (p >= 0.5).astype(int)
        cm = confusion_matrix(test_labels, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"]).plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=200)
    plt.close()

    # --- Ensemble uncertainty vs. correctness (Section 3.6) ---
    std_path = os.path.join(RESULTS_DIR, "ensemble_test_probs_std.npy")
    mean_path = os.path.join(RESULTS_DIR, "ensemble_test_probs.npy")
    if os.path.exists(std_path) and os.path.exists(mean_path):
        std_prob = np.load(std_path)
        mean_prob = np.load(mean_path)
        pred = (mean_prob >= 0.5).astype(int)
        correct = (pred == test_labels.astype(int))
        plt.figure(figsize=(5, 4))
        plt.boxplot([std_prob[correct], std_prob[~correct]], labels=["Correct", "Incorrect"])
        plt.ylabel("Ensemble std. dev. (uncertainty)")
        plt.title("Ensemble disagreement vs. prediction correctness")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "ensemble_uncertainty_vs_correctness.png"), dpi=200)
        plt.close()

    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
