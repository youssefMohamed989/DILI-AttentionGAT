"""
13_advanced_attribution_figure.py
Figure 12 and the graphical abstract: a composite, publication-grade
attention-attribution figure with a continuous colormap, bond
highlighting, and a shared colorbar across a grid of representative
molecules.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from rdkit.Chem.Draw import rdMolDraw2D

from common import CHECKPOINT_DIR, FIGURES_DIR
from dataset_io import load_graphs_and_labels
from gnn_model import DILI_GAT, collate_graphs


def normalize_per_molecule(alpha: np.ndarray, graph_index: np.ndarray) -> np.ndarray:
    """min-max normalize attention weights to [0, 1] within each molecule
    (same convention as 05_interpretability.py)."""
    out = np.zeros_like(alpha)
    for g in np.unique(graph_index):
        mask = graph_index == g
        vals = alpha[mask]
        lo, hi = vals.min(), vals.max()
        out[mask] = (vals - lo) / (hi - lo) if hi > lo else 0.5
    return out


def render_panel(mol, weights, colormap=cm.Reds):
    atom_colors = {i: tuple(colormap(w)[:3]) for i, w in enumerate(weights)}
    bond_colors = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = (weights[i] + weights[j]) / 2.0
        bond_colors[bond.GetIdx()] = tuple(colormap(w)[:3])
    drawer = rdMolDraw2D.MolDraw2DCairo(350, 350)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors.keys()), highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    tmp_path = "/tmp/_panel.png"
    with open(tmp_path, "wb") as f:
        f.write(png_bytes)
    return plt.imread(tmp_path)


def main():
    test_graphs, test_labels, test_df = load_graphs_and_labels("test")
    model = DILI_GAT()
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "gat_single_model.pt"),
                                      map_location="cpu"))
    model.eval()

    batch = collate_graphs(test_graphs)
    with torch.no_grad():
        logit, alpha = model(batch, return_attention=True)
    prob = torch.sigmoid(logit).numpy()
    graph_index = batch.graph_index.numpy()
    alpha_norm = normalize_per_molecule(alpha.numpy(), graph_index)

    n_panels = min(6, len(test_graphs))
    conf = np.abs(prob - 0.5)
    chosen = np.argsort(-conf)[:n_panels]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, idx in zip(axes.flat, chosen):
        mol = test_graphs[idx]["mol"]
        mask = graph_index == idx
        weights = alpha_norm[mask]
        img = render_panel(mol, weights)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"pred={prob[idx]:.2f} true={int(test_labels[idx])}", fontsize=9)

    sm = cm.ScalarMappable(cmap=cm.Reds, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="normalized attention weight")
    fig.suptitle("Figure 12. Attention-attribution maps, representative test molecules")
    plt.savefig(os.path.join(FIGURES_DIR, "figure12_attribution_composite.png"), dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"Wrote figure12_attribution_composite.png and graphical abstract source panels to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
