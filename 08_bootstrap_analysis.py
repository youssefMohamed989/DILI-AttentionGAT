"""
08_bootstrap_analysis.py
Section 2.8 / Table 3. Nonparametric percentile bootstrap over the n=108
primary-split test-set predictions (5,000 resamples): 95% CI for each
model's AUROC, plus paired bootstrap significance tests comparing the
deep-ensemble GAT to each baseline.
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from common import RESULTS_DIR, N_BOOTSTRAP, SEED
from dataset_io import load_graphs_and_labels


def bootstrap_auroc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aurocs = np.array(aurocs)
    lo, hi = np.percentile(aurocs, [2.5, 97.5])
    return float(lo), float(hi), aurocs


def paired_bootstrap_pvalue(y_true, prob_a, prob_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Two-sided paired bootstrap test: fraction of resamples where the sign
    of (AUROC_a - AUROC_b) flips relative to the observed sign."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    observed_diff = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        d = roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        diffs.append(d)
    diffs = np.array(diffs)
    if observed_diff >= 0:
        p = 2 * min((diffs < 0).mean(), 0.5)
    else:
        p = 2 * min((diffs > 0).mean(), 0.5)
    return float(min(p, 1.0)), diffs


def main():
    _, test_labels, _ = load_graphs_and_labels("test")

    model_files = {
        "GAT, single model": "gat_single_test_probs.npy",
        "GAT, 5-model deep ensemble": "ensemble_test_probs.npy",
        "RF + Morgan FP": "rf_morgan_test_probs.npy",
        "LR + Morgan FP": "lr_morgan_test_probs.npy",
        "RF + RDKit descriptors": "rf_descriptors_test_probs.npy",
    }

    probs = {}
    for name, fname in model_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            probs[name] = np.load(path)

    table3 = {}
    for name, p in probs.items():
        lo, hi, _ = bootstrap_auroc_ci(test_labels, p)
        table3[name] = {
            "auroc_point_estimate": float(roc_auc_score(test_labels, p)),
            "ci_95_low": lo, "ci_95_high": hi,
        }

    significance = {}
    if "GAT, 5-model deep ensemble" in probs:
        ref = probs["GAT, 5-model deep ensemble"]
        for name, p in probs.items():
            if name == "GAT, 5-model deep ensemble":
                continue
            pval, _ = paired_bootstrap_pvalue(test_labels, ref, p)
            significance[f"ensemble_vs_{name}"] = {"p_value": pval}

    out = {"table3_bootstrap_ci": table3, "paired_significance_vs_ensemble": significance}
    with open(os.path.join(RESULTS_DIR, "table3_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
