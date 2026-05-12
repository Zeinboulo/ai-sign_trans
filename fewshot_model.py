from typing import Optional, Tuple

import torch
from torch import nn


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

    # Left hand: 0-20
    for a, b in HAND_CONNECTIONS:
        adj[a, b] = 1
        adj[b, a] = 1

    # Right hand: 21-41
    offset = 21
    for a, b in HAND_CONNECTIONS:
        adj[offset + a, offset + b] = 1
        adj[offset + b, offset + a] = 1

    # Pose: 42-74
    offset = 42
    for a, b in POSE_CONNECTIONS:
        adj[offset + a, offset + b] = 1
        adj[offset + b, offset + a] = 1

    # Self loops
    adj = adj + torch.eye(num_nodes)

    # Row-normalize
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj = adj / deg
    return adj


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STGCNEncoder(nn.Module):
    def __init__(self, num_nodes: int = 75, in_channels: int = 3, hidden: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        A = build_adjacency(num_nodes)
        self.num_nodes = num_nodes
        self.block1 = STGCNBlock(in_channels, hidden, A)
        self.block2 = STGCNBlock(hidden, out_dim, A)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, 225) -> (B, C, T, V)
        b, t, f = x.shape
        v = self.num_nodes
        x = x.view(b, t, v, 3).permute(0, 3, 1, 2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)

        if lengths is not None:
            lengths = lengths.to(x.device).clamp(min=1, max=t)
            mask = torch.arange(t, device=x.device).view(1, 1, t, 1) < lengths.view(-1, 1, 1, 1)
            x = x * mask
            denom = lengths.float().view(-1, 1) * v
        else:
            denom = torch.tensor(float(t * v), device=x.device).view(1, 1)

        x = x.sum(dim=(2, 3)) / denom
        return x


class ProtoNet(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        support_x: torch.Tensor,
        support_len: torch.Tensor,
        query_x: torch.Tensor,
        query_len: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        emb_support = self.encoder(support_x, support_len)
        prototypes = emb_support.view(n_way, k_shot, -1).mean(dim=1)

        emb_query = self.encoder(query_x, query_len)
        dists = torch.cdist(emb_query, prototypes)
        logits = -dists
        return logits
