"""
gnn_model_v3.py
Chemistry-informed multi-task GAT (GAT-CI-MT), Section 2.13. Builds on
DILI_GAT_JK, adding a second auxiliary output head that predicts the
multi-label presence/absence of the seven structural alerts from the same
pooled molecular embedding used for DILI classification. Trained with
combined loss L = L_DILI + lambda * L_toxicophore, lambda = 0.4 (fixed a
priori, not tuned on the test set).
"""
import torch
import torch.nn as nn

from common import (
    ATOM_FEATURE_DIM, GAT_HIDDEN_DIM, GAT_NUM_LAYERS, GAT_NUM_HEADS, GAT_DROPOUT,
    N_TOXICOPHORES,
)
from gnn_model import GraphAttentionLayer, AttentionPooling, MolGraphBatch


class DILI_GAT_CI_MT(nn.Module):
    """GAT+JK backbone (identical to gnn_model_v2.DILI_GAT_JK) with an
    added auxiliary multi-label toxicophore-presence head sharing the
    pooled molecular embedding."""

    def __init__(self, in_dim: int = ATOM_FEATURE_DIM, hidden_dim: int = GAT_HIDDEN_DIM,
                 n_layers: int = GAT_NUM_LAYERS, n_heads: int = GAT_NUM_HEADS,
                 dropout: float = GAT_DROPOUT, n_aux_labels: int = N_TOXICOPHORES):
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
        # auxiliary structural-alert-recognition head (multi-label)
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_aux_labels),
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

    def forward_multitask(self, batch: MolGraphBatch):
        x = self.encode_atoms(batch)
        pooled, alpha = self.pool(x, batch.graph_index, batch.num_graphs)
        dili_logit = self.classifier(pooled).squeeze(-1)
        aux_logits = self.aux_head(pooled)
        return dili_logit, alpha, aux_logits

    def forward(self, batch: MolGraphBatch, return_attention: bool = False):
        dili_logit, alpha, _ = self.forward_multitask(batch)
        if return_attention:
            return dili_logit, alpha
        return dili_logit
