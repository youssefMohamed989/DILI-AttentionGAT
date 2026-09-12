"""
10_repeated_cv_figure_and_litcomparison.py
Section 3.7 / Table 6. Visualizes per-split test AUROC for the
single-model GAT+JK vs. the RF/Morgan baseline across the five
independent scaffold splits (07_repeated_scaffold_cv.py output), and
reports a paired t-test on the per-split AUROC values.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import RESULTS_DIR, FIGURES_DIR


def main():
    with open(os.path.join(RESULTS_DIR, "repeated_cv_single_model.json")) as f:
        summary = json.load(f)

    per_split = summary["per_split"]
    gat = np.array([r["gat_jk_test_auroc"] for r in per_split])
    rf = np.array([r["rf_morgan_test_auroc"] for r in per_split])

    t_stat, p_value = stats.ttest_rel(gat, rf)
    lit_comparison = {
        "gat_jk_mean": float(gat.mean()), "gat_jk_std": float(gat.std()),
        "rf_morgan_mean": float(rf.mean()), "rf_morgan_std": float(rf.std()),
        "paired_t_statistic": float(t_stat), "paired_t_pvalue": float(p_value),
    }
    with open(os.path.join(RESULTS_DIR, "repeated_cv_ttest.json"), "w") as f:
        json.dump(lit_comparison, f, indent=2)
    print(json.dumps(lit_comparison, indent=2))

    x = np.arange(len(per_split))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, gat, width, label="GAT+JK, single model")
    plt.bar(x + width / 2, rf, width, label="RF + Morgan FP")
    plt.xticks(x, [f"split {i+1}" for i in x])
    plt.ylabel("Test AUROC")
    plt.title(f"Repeated scaffold-split CV (paired t-test p={p_value:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "repeated_cv_comparison.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
