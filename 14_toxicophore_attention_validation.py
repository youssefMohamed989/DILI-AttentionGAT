"""
14_toxicophore_attention_validation.py
Section 2.12 / Table 7 / Figure 13. For every compound in the full curated
dataset, flags which atoms belong to each of the seven literature-derived
structural alerts (common.TOXICOPHORE_SMARTS), min-max normalizes the
trained single-model GAT's attention-pooling weights within each molecule,
and compares in-alert vs. out-of-alert attention pooled across the full
dataset via a one-sided Mann-Whitney U test (per alert and pooled across
all alerts). Also tests whether alert count correlates with the true DILI
label (point-biserial) and predicted DILI probability (Pearson).
"""
import json
import os

import numpy as np
import torch
from scipy import stats

from common import RESULTS_DIR, CHECKPOINT_DIR, TOXICOPHORE_NAMES, DISPLAY_NAMES
from dataset_io import (
    load_partition, dataframe_to_graphs, toxicophore_labels, toxicophore_atom_flags,
)
from gnn_model import DILI_GAT, collate_graphs


def normalize_per_molecule(alpha, graph_index):
    out = np.zeros_like(alpha)
    for g in np.unique(graph_index):
        mask = graph_index == g
        vals = alpha[mask]
        lo, hi = vals.min(), vals.max()
        out[mask] = (vals - lo) / (hi - lo) if hi > lo else 0.5
    return out


def run_validation(model, full_df, device="cpu"):
    graphs, keep_mask = dataframe_to_graphs(full_df)
    df = full_df.loc[keep_mask].reset_index(drop=True)
    labels = df["DILI_label"].values.astype(int)

    model.eval()
    batch = collate_graphs(graphs).to(device)
    with torch.no_grad():
        logit, alpha = model(batch, return_attention=True)
    prob = torch.sigmoid(logit).cpu().numpy()
    alpha = alpha.cpu().numpy()
    graph_index = batch.graph_index.cpu().numpy()
    alpha_norm = normalize_per_molecule(alpha, graph_index)

    alert_mol_labels = toxicophore_labels(df)  # (n_mols, 7)

    # per-alert atom-level pooling of in-alert vs out-of-alert attention,
    # across every molecule that contains at least one atom of that alert
    per_alert_in, per_alert_out = {n: [] for n in TOXICOPHORE_NAMES}, {n: [] for n in TOXICOPHORE_NAMES}
    pooled_in_all, pooled_out_all = [], []

    for gi, g in enumerate(graphs):
        mol = g["mol"]
        mask = graph_index == gi
        weights = alpha_norm[mask]
        atom_flags = toxicophore_atom_flags(mol)  # (n_atoms, 7)
        any_alert_atom = atom_flags.any(axis=1)
        pooled_in_all.extend(weights[any_alert_atom].tolist())
        pooled_out_all.extend(weights[~any_alert_atom].tolist())
        for j, name in enumerate(TOXICOPHORE_NAMES):
            col = atom_flags[:, j]
            if col.any():
                per_alert_in[name].extend(weights[col].tolist())
                per_alert_out[name].extend(weights[~col].tolist())

    table7 = {}
    for name in TOXICOPHORE_NAMES:
        in_vals, out_vals = np.array(per_alert_in[name]), np.array(per_alert_out[name])
        if len(in_vals) == 0 or len(out_vals) == 0:
            continue
        u_stat, p_val = stats.mannwhitneyu(in_vals, out_vals, alternative="greater")
        enrichment = (in_vals.mean() / out_vals.mean()) if out_vals.mean() > 0 else float("nan")
        table7[name] = {
            "display_name": DISPLAY_NAMES[name],
            "mean_attention_in_alert": float(in_vals.mean()),
            "mean_attention_out_alert": float(out_vals.mean()),
            "enrichment_ratio": float(enrichment),
            "mannwhitney_p_onesided": float(p_val),
            "n_atoms_in_alert": int(len(in_vals)),
        }

    pooled_in_all, pooled_out_all = np.array(pooled_in_all), np.array(pooled_out_all)
    u_stat, pooled_p = stats.mannwhitneyu(pooled_in_all, pooled_out_all, alternative="greater")
    pooled_enrichment = pooled_in_all.mean() / pooled_out_all.mean() if pooled_out_all.mean() > 0 else float("nan")

    alert_counts = alert_mol_labels.sum(axis=1)
    point_biserial = stats.pointbiserialr(labels, alert_counts)
    pearson = stats.pearsonr(alert_counts, prob)

    summary = {
        "table7_per_alert": table7,
        "pooled_enrichment_ratio": float(pooled_enrichment),
        "pooled_mannwhitney_p_onesided": float(pooled_p),
        "alert_count_vs_dili_label_point_biserial_r": float(point_biserial.correlation),
        "alert_count_vs_dili_label_pvalue": float(point_biserial.pvalue),
        "alert_count_vs_predicted_prob_pearson_r": float(pearson[0]),
        "alert_count_vs_predicted_prob_pvalue": float(pearson[1]),
    }
    return summary, prob, alert_mol_labels, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from repeated_split_utils import full_curated_dataframe
    full_df = full_curated_dataframe()

    model = DILI_GAT()
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gat_single_model.pt"),
                                      map_location="cpu"))

    summary, prob, alert_labels, dili_labels = run_validation(model, full_df, device=device)
    with open(os.path.join(RESULTS_DIR, "table7_toxicophore_validation_baseline_gat.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Figure 13: per-alert enrichment bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import FIGURES_DIR

    names = list(summary["table7_per_alert"].keys())
    enrich = [summary["table7_per_alert"][n]["enrichment_ratio"] for n in names]
    labels_disp = [summary["table7_per_alert"][n]["display_name"] for n in names]
    plt.figure(figsize=(7, 4))
    plt.barh(labels_disp, enrich)
    plt.axvline(1.0, color="gray", linestyle="--")
    plt.xlabel("Attention enrichment ratio (in-alert / out-of-alert)")
    plt.title("Figure 13. Baseline GAT toxicophore-attention enrichment")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure13_toxicophore_enrichment_baseline.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
