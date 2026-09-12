"""
gnn_model.py
Full PyTorch implementation (Section 2.3) of:
  - GraphAttentionLayer: a single multi-head, edge-conditioned graph
    attention message-passing layer with residual connection + LayerNorm.
  - AttentionPooling: learned attention-pooling readout used both for
    producing the molecular embedding and for the substructure attribution
    analysis (Section 3.8 / 2.12).
  - DILI_GAT: the 3-layer single-model GAT classifier (Table 2, "GAT,
    single model").
  - MolGraphBatch / collate_graphs: sparse batching utilities for
    variable-size molecular graphs (no external graph-learning framework
    is used, per Section 2.3).

Written from scratch in plain PyTorch, deliberately avoiding PyTorch
Geometric/DGL, so the full message-passing and attention logic is
auditable directly from this file.
"""
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, GAT_NUM_HEADS, GAT_HIDDEN_DIM, GAT_DROPOUT


# ---------------------------------------------------------------------------
# Sparse batching
# ---------------------------------------------------------------------------
@dataclass
class MolGraphBatch:
    node_feats: torch.Tensor      # (N_total, ATOM_FEATURE_DIM)
    edge_index: torch.Tensor      # (2, E_total)
    edge_feats: torch.Tensor      # (E_total, BOND_FEATURE_DIM)
    graph_index: torch.Tensor     # (N_total,) which graph each node belongs to
    num_graphs: int
    labels: torch.Tensor = None           # (num_graphs,)
    aux_labels: torch.Tensor = None       # (num_graphs, n_aux) optional, chem-informed model

    def to(self, device):
        self.node_feats = self.node_feats.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_feats = self.edge_feats.to(device)
        self.graph_index = self.graph_index.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.aux_labels is not None:
            self.aux_labels = self.aux_labels.to(device)
        return self


def collate_graphs(graphs: List[dict], labels=None, aux_labels=None) -> MolGraphBatch:
    """Combine a list of single-molecule graph dicts (from graph_features.mol_to_graph)
    into one block-diagonal sparse batch, PyG-style."""
    node_feats, edge_index_list, edge_feats, graph_index = [], [], [], []
    node_offset = 0
    for gi, g in enumerate(graphs):
        n = g["node_feats"].shape[0]
        node_feats.append(torch.as_tensor(g["node_feats"], dtype=torch.float32))
        ei = torch.as_tensor(g["edge_index"], dtype=torch.long) + node_offset
        edge_index_list.append(ei)
        edge_feats.append(torch.as_tensor(g["edge_feats"], dtype=torch.float32))
        graph_index.append(torch.full((n,), gi, dtype=torch.long))
        node_offset += n

    batch = MolGraphBatch(
        node_feats=torch.cat(node_feats, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_feats=torch.cat(edge_feats, dim=0),
        graph_index=torch.cat(graph_index, dim=0),
        num_graphs=len(graphs),
    )
    if labels is not None:
        batch.labels = torch.as_tensor(labels, dtype=torch.float32)
    if aux_labels is not None:
        batch.aux_labels = torch.as_tensor(aux_labels, dtype=torch.float32)
    return batch


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax of `scores` grouped by `index` (e.g. per destination node, or
    per graph). Numerically stable via a scatter-max subtraction."""
    seg_max = torch.full((num_segments,), float("-inf"), device=scores.device, dtype=scores.dtype)
    seg_max = seg_max.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    seg_max = torch.nan_to_num(seg_max, neginf=0.0)
    scores = scores - seg_max[index]
    exp_scores = torch.exp(scores)
    seg_sum = torch.zeros((num_segments,), device=scores.device, dtype=scores.dtype)
    seg_sum = seg_sum.scatter_add(0, index, exp_scores)
    seg_sum = seg_sum.clamp_min(1e-12)
    return exp_scores / seg_sum[index]


# ---------------------------------------------------------------------------
# Graph attention layer
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """One multi-head, edge-conditioned graph-attention message-passing
    layer (Section 2.3). For every directed edge (src -> dst):
      msg_src  = W_src * x_src + W_edge * e_(src,dst)
      score    = LeakyReLU( a^T [W_dst * x_dst || msg_src] )
      alpha    = softmax_over_incoming_edges(score)
    Messages are aggregated by attention-weighted sum, projected per head,
    then combined with the previous layer's representation via a residual
    connection and LayerNorm.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = GAT_NUM_HEADS,
                 edge_dim: int = BOND_FEATURE_DIM, dropout: float = GAT_DROPOUT,
                 negative_slope: float = 0.2):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim

        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)

        # per-head attention vector, applied to [dst || msg] concat (2*head_dim)
        self.attn = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feats: torch.Tensor):
        n_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        h_src = self.W_src(x)              # (N, out_dim)
        h_dst = self.W_dst(x)              # (N, out_dim)
        h_edge = self.W_edge(edge_feats)    # (E, out_dim)

        msg = h_src[src] + h_edge          # edge-conditioned source message, (E, out_dim)
        dst_feat = h_dst[dst]               # (E, out_dim)

        H, D = self.n_heads, self.head_dim
        msg_h = msg.view(-1, H, D)
        dst_h = dst_feat.view(-1, H, D)
        cat_h = torch.cat([dst_h, msg_h], dim=-1)              # (E, H, 2D)
        e = self.leaky_relu((cat_h * self.attn.unsqueeze(0)).sum(-1))  # (E, H)

        # softmax normalized per destination node, independently per head
        alpha = torch.stack(
            [segment_softmax(e[:, h], dst, n_nodes) for h in range(H)], dim=-1
        )  # (E, H)
        alpha = self.dropout(alpha)

        weighted_msg = msg_h * alpha.unsqueeze(-1)  # (E, H, D)
        weighted_msg = weighted_msg.view(-1, self.out_dim)

        agg = torch.zeros((n_nodes, self.out_dim), device=x.device, dtype=x.dtype)
        agg = agg.index_add(0, dst, weighted_msg)

        out = self.out_proj(agg)
        out = out + self.residual_proj(x)
        out = self.norm(out)
        return out


# ---------------------------------------------------------------------------
# Attention-pooling readout
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Learned attention-pooling readout: a linear scoring function assigns
    each atom a scalar importance score, softmax-normalized over the atoms
    of that molecule, and used to compute a weighted sum of atom
    embeddings. Returns both the pooled embedding and the per-atom
    attention weights (used directly for the interpretability analysis)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, graph_index: torch.Tensor, num_graphs: int):
        raw_scores = self.score(x).squeeze(-1)  # (N,)
        alpha = segment_softmax(raw_scores, graph_index, num_graphs)  # (N,), per-molecule softmax
        weighted = x * alpha.unsqueeze(-1)
        pooled = torch.zeros((num_graphs, x.shape[1]), device=x.device, dtype=x.dtype)
        pooled = pooled.index_add(0, graph_index, weighted)
        return pooled, alpha


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class DILI_GAT(nn.Module):
    """3-layer single-model graph attention network for DILI classification
    (Table 2, 'GAT, single model')."""

    def __init__(self, in_dim: int = ATOM_FEATURE_DIM, hidden_dim: int = GAT_HIDDEN_DIM,
                 n_layers: int = 3, n_heads: int = GAT_NUM_HEADS, dropout: float = GAT_DROPOUT):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pool = AttentionPooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_atoms(self, batch: MolGraphBatch) -> torch.Tensor:
        x = self.embed(batch.node_feats)
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_feats)
        return x

    def forward(self, batch: MolGraphBatch, return_attention: bool = False):
        x = self.encode_atoms(batch)
        pooled, alpha = self.pool(x, batch.graph_index, batch.num_graphs)
        logit = self.classifier(pooled).squeeze(-1)
        if return_attention:
            return logit, alpha
        return logit
