import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
]


def build_adjacency(num_nodes: int = 75) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for a, b in HAND_CONNECTIONS:
        adj[a, b] = 1
        adj[b, a] = 1

    offset = 21
    for a, b in HAND_CONNECTIONS:
        adj[offset + a, offset + b] = 1
        adj[offset + b, offset + a] = 1

    offset = 42
    for a, b in POSE_CONNECTIONS:
        adj[offset + a, offset + b] = 1
        adj[offset + b, offset + a] = 1

    adj = adj + torch.eye(num_nodes)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        return self.relu(x)


class GraphBackbone(nn.Module):
    def __init__(
        self,
        num_nodes: int = 75,
        in_channels: int = 3,
        hidden: int = 64,
        out_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        A = build_adjacency(num_nodes)
        self.num_nodes = num_nodes
        self.block1 = STGCNBlock(in_channels, hidden, A)
        self.block2 = STGCNBlock(hidden, out_dim, A)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        v = self.num_nodes
        x = x.view(b, t, v, 3).permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        return self.dropout(x)


class AttnPool1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale = math.sqrt(x.size(-1))
        scores = (x * self.query).sum(dim=-1) / scale
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bt,btd->bd", weights, x)


class HybridGraphTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        max_len: int,
        num_nodes: int = 75,
        in_channels: int = 3,
        gcn_hidden: int = 64,
        gcn_out: int = 128,
        tf_layers: int = 4,
        tf_heads: int = 8,
        tf_ff: int = 256,
        dropout: float = 0.2,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.backbone = GraphBackbone(num_nodes, in_channels, gcn_hidden, gcn_out, dropout)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, gcn_out))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gcn_out,
            nhead=tf_heads,
            dim_feedforward=tf_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.attn_pool = AttnPool1D(gcn_out)
        self.norm = nn.LayerNorm(gcn_out)

        self.projector = nn.Sequential(
            nn.Linear(gcn_out, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(gcn_out, gcn_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_out // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.backbone(x)
        x = x.mean(dim=3).permute(0, 2, 1)
        t = x.size(1)

        if t == self.max_len:
            pos = self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=t,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        x = self.pos_drop(x + pos)

        pad_mask = None
        if lengths is not None:
            lengths = lengths.to(x.device).clamp(min=1, max=t)
            pad_mask = torch.arange(t, device=x.device).view(1, -1) >= lengths.view(-1, 1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        pooled = self.attn_pool(x, pad_mask)
        emb = self.norm(pooled)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, self.projector(emb)
        return logits, None
