"""
gnn_model_v2.py
Advanced GAT+JK (jumping-knowledge) architecture (Section 2.3, final
architecture), a reusable single-model training routine shared by the
ensemble/repeated-CV scripts, and deep-ensemble prediction/averaging
utilities (Section 2.4).
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import (
    ATOM_FEATURE_DIM, GAT_HIDDEN_DIM, GAT_NUM_HEADS, GAT_NUM_LAYERS, GAT_DROPOUT,
    LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP_NORM, EARLY_STOP_PATIENCE, MAX_EPOCHS,
    BATCH_SIZE,
)
from gnn_model import GraphAttentionLayer, AttentionPooling, MolGraphBatch


class DILI_GAT_JK(nn.Module):
    """GAT+JK: concatenates the atom representations produced after every
    message-passing layer (including the initial embedding) -- a
    jumping-knowledge connection -- and projects the concatenation back to
    the working hidden dimension before the attention-pooling readout."""

    def __init__(self, in_dim: int = ATOM_FEATURE_DIM, hidden_dim: int = GAT_HIDDEN_DIM,
                 n_layers: int = GAT_NUM_LAYERS, n_heads: int = GAT_NUM_HEADS,
                 dropout: float = GAT_DROPOUT):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.jk_proj = nn.Linear(hidden_dim * (n_layers + 1), hidden_dim)
        self.pool = AttentionPooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_atoms(self, batch: MolGraphBatch) -> torch.Tensor:
        x0 = self.embed(batch.node_feats)
        layer_outputs = [x0]
        x = x0
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_feats)
            layer_outputs.append(x)
        jk = torch.cat(layer_outputs, dim=-1)
        return self.jk_proj(jk)

    def forward(self, batch: MolGraphBatch, return_attention: bool = False):
        x = self.encode_atoms(batch)
        pooled, alpha = self.pool(x, batch.graph_index, batch.num_graphs)
        logit = self.classifier(pooled).squeeze(-1)
        if return_attention:
            return logit, alpha
        return logit


# ---------------------------------------------------------------------------
# Shared training routine (used by 02/06/07/11/15/17)
# ---------------------------------------------------------------------------
def compute_pos_weight(train_labels: np.ndarray) -> float:
    """positive class weight = training-set negative:positive ratio."""
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    return n_neg / max(n_pos, 1.0)


def train_single_model(
    model: nn.Module,
    train_graphs: List[dict], train_labels: np.ndarray,
    val_graphs: List[dict], val_labels: np.ndarray,
    collate_fn,
    device: str = "cpu",
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    verbose: bool = True,
    aux_train_labels: Optional[np.ndarray] = None,
    aux_val_labels: Optional[np.ndarray] = None,
    aux_lambda: float = 0.0,
):
    """Trains `model` with Adam + class-weighted BCE, gradient clipping,
    and early stopping on validation AUROC (Section 2.6). If aux_lambda > 0
    and aux_*_labels are given, adds the multi-task toxicophore loss
    (Section 2.13) and expects `model.forward` to accept
    `return_attention=True` returning (logit, alpha) with alpha usable
    to recover a pooled embedding is not required here -- aux head, when
    present, must be exposed as `model.aux_head` operating on the same
    pooled embedding as the classifier (see gnn_model_v3.DILI_GAT_CI_MT).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = torch.tensor(compute_pos_weight(train_labels), device=device)
    dili_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_criterion = nn.BCEWithLogitsLoss()

    n_train = len(train_graphs)
    best_val_auroc = -1.0
    best_state = None
    epochs_since_improve = 0
    history = {"train_loss": [], "val_auroc": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            graphs = [train_graphs[i] for i in idx]
            labels = train_labels[idx]
            aux = aux_train_labels[idx] if aux_train_labels is not None else None
            batch = collate_fn(graphs, labels=labels, aux_labels=aux).to(device)

            optimizer.zero_grad()
            if aux_lambda > 0 and hasattr(model, "aux_head"):
                logit, alpha, aux_logits = model.forward_multitask(batch)
                loss = dili_criterion(logit, batch.labels) + aux_lambda * aux_criterion(
                    aux_logits, batch.aux_labels
                )
            else:
                logit = model(batch)
                loss = dili_criterion(logit, batch.labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n_train

        # validation
        model.eval()
        with torch.no_grad():
            val_batch = collate_fn(val_graphs, labels=val_labels).to(device)
            if aux_lambda > 0 and hasattr(model, "aux_head"):
                val_logit, _, _ = model.forward_multitask(val_batch)
            else:
                val_logit = model(val_batch)
            val_prob = torch.sigmoid(val_logit).cpu().numpy()
        try:
            val_auroc = roc_auc_score(val_labels, val_prob)
        except ValueError:
            val_auroc = float("nan")

        history["train_loss"].append(epoch_loss)
        history["val_auroc"].append(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | train_loss {epoch_loss:.4f} | val_auroc {val_auroc:.4f}")

        if epochs_since_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val AUROC={best_val_auroc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_auroc, history


@torch.no_grad()
def predict_proba(model: nn.Module, graphs: List[dict], collate_fn, device: str = "cpu",
                   return_attention: bool = False):
    model.eval()
    batch = collate_fn(graphs).to(device)
    if return_attention:
        logit, alpha = model(batch, return_attention=True)
        return torch.sigmoid(logit).cpu().numpy(), alpha.cpu().numpy(), batch.graph_index.cpu().numpy()
    logit = model(batch)
    return torch.sigmoid(logit).cpu().numpy()


class DeepEnsemble:
    """A 5-member deep ensemble of DILI_GAT_JK models (Section 2.4). Members
    differ only in the random seed governing weight initialization and
    mini-batch shuffling; predictions are averaged, and the cross-member
    standard deviation is retained as a per-molecule uncertainty score."""

    def __init__(self, models: List[nn.Module]):
        self.models = models

    @torch.no_grad()
    def predict(self, graphs: List[dict], collate_fn, device: str = "cpu"):
        all_probs = []
        for m in self.models:
            m.eval()
            batch = collate_fn(graphs).to(device)
            logit = m(batch)
            all_probs.append(torch.sigmoid(logit).cpu().numpy())
        all_probs = np.stack(all_probs, axis=0)  # (n_members, n_mols)
        mean_prob = all_probs.mean(axis=0)
        std_prob = all_probs.std(axis=0)
        return mean_prob, std_prob, all_probs
