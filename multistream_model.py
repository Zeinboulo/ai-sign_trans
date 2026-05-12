from typing import Optional, Tuple
import math

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

LEFT_HAND_IDX = list(range(0, 21))
RIGHT_HAND_IDX = list(range(21, 42))
POSE_IDX = list(range(42, 75))

FULL_CONNECTIONS = (
    HAND_CONNECTIONS
    + [(a + 21, b + 21) for a, b in HAND_CONNECTIONS]
    + [(a + 42, b + 42) for a, b in POSE_CONNECTIONS]
)


def build_adjacency_matrices(
    num_nodes: int, connections: list[tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for a, b in connections:
        adj[a, b] = 1
        adj[b, a] = 1
    adj = adj + torch.eye(num_nodes)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj_norm = adj / deg

    adj2 = (adj @ adj)
    adj2 = (adj2 > 0).float()
    deg2 = adj2.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj2_norm = adj2 / deg2
    return adj_norm, adj2_norm


class STCAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)

        t = x.mean(dim=3)
        t = self.temporal(t).unsqueeze(3)
        x = x * t

        s = x.mean(dim=2)
        s = self.spatial(s).unsqueeze(2)
        x = x * s
        return x


class AdaptiveGraphConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A1: torch.Tensor,
        A2: torch.Tensor,
        groups: int = 4,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        self.register_buffer("A1", A1)
        self.register_buffer("A2", A2)
        self.groups = groups
        self.num_nodes = A1.size(0)
        self.inter_channels = max(8, in_channels // (groups * 2))

        self.theta = nn.Conv1d(
            in_channels,
            self.inter_channels * groups,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.phi = nn.Conv1d(
            in_channels,
            self.inter_channels * groups,
            kernel_size=1,
            groups=groups,
            bias=False,
        )

        self.A_global = nn.Parameter(torch.zeros(groups, self.num_nodes, self.num_nodes))
        self.A_global2 = nn.Parameter(torch.zeros(groups, self.num_nodes, self.num_nodes))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v = x.shape
        x_mean = x.mean(dim=2)
        theta = self.theta(x_mean).view(b, self.groups, self.inter_channels, v)
        phi = self.phi(x_mean).view(b, self.groups, self.inter_channels, v)
        attn = torch.einsum("bgcv,bgcw->bgvw", theta, phi) / math.sqrt(self.inter_channels)
        attn = torch.softmax(attn, dim=-1)

        x = x.view(b, self.groups, c // self.groups, t, v)
        out = []
        for g in range(self.groups):
            A1 = self.A1 + self.A_global[g] + attn[:, g]
            A2 = self.A2 + self.A_global2[g]
            xg = x[:, g]
            x1 = torch.einsum("bctv,bvw->bctw", xg, A1)
            x2 = torch.einsum("bctv,vw->bctw", xg, A2)
            out.append(x1 + x2)
        x = torch.cat(out, dim=1)
        return self.conv(x)


class AdaptiveSTGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A1: torch.Tensor,
        A2: torch.Tensor,
        groups: int = 4,
    ) -> None:
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A1, A2, groups=groups)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.attn = STCAttention(out_channels)
        self.relu = nn.ReLU()
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = self.attn(x)
        return self.relu(x + res)


class StreamEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        connections: list[tuple[int, int]],
        in_channels: int = 3,
        hidden: int = 64,
        out_dim: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        A1, A2 = build_adjacency_matrices(num_nodes, connections)
        self.num_nodes = num_nodes
        self.block1 = AdaptiveSTGCNBlock(in_channels, hidden, A1, A2, groups=1)
        self.block2 = AdaptiveSTGCNBlock(hidden, out_dim, A1, A2, groups=4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = res + self.drop1(attn_out)

        res = x
        x = self.norm2(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = res + self.drop2(x)

        res = x
        x = self.norm3(x)
        x = self.ff(x)
        x = res + self.drop3(x)
        return x


class AttnPool1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        scores = (x * self.query).sum(dim=-1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bt,btd->bd", weights, x)


class MultiStreamEncoder(nn.Module):
    def __init__(
        self,
        max_len: int,
        gcn_hidden: int = 64,
        gcn_out: int = 96,
        d_model: int = 192,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        conformer_ff: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.num_nodes = 75
        self.joint = StreamEncoder(self.num_nodes, FULL_CONNECTIONS, hidden=gcn_hidden, out_dim=gcn_out, dropout=dropout)
        self.bone = StreamEncoder(self.num_nodes, FULL_CONNECTIONS, hidden=gcn_hidden, out_dim=gcn_out, dropout=dropout)
        self.motion = StreamEncoder(self.num_nodes, FULL_CONNECTIONS, hidden=gcn_hidden, out_dim=gcn_out, dropout=dropout)
        self.bone_pairs = FULL_CONNECTIONS
        counts = torch.zeros(self.num_nodes, dtype=torch.float32)
        for a, b in self.bone_pairs:
            counts[a] += 1
            counts[b] += 1
        self.register_buffer("bone_counts", counts.view(1, 1, -1, 1))

        self.fuse = nn.Sequential(
            nn.Linear(gcn_out * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, conformer_heads, conformer_ff, dropout=dropout) for _ in range(conformer_layers)]
        )
        self.pool = AttnPool1D(d_model)
        self.norm = nn.LayerNorm(d_model)

    def _compute_bone(self, joints: torch.Tensor) -> torch.Tensor:
        bone = torch.zeros_like(joints)
        for a, b in self.bone_pairs:
            diff = joints[:, :, b, :] - joints[:, :, a, :]
            bone[:, :, b, :] += diff
            bone[:, :, a, :] -= diff
        return bone / self.bone_counts.clamp(min=1.0)

    def _compute_motion(self, joints: torch.Tensor) -> torch.Tensor:
        motion = torch.zeros_like(joints)
        motion[:, 1:] = joints[:, 1:] - joints[:, :-1]
        return motion

    def encode_sequence(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t, f = x.shape
        joints = x.view(b, t, self.num_nodes, 3)
        bones = self._compute_bone(joints)
        motion = self._compute_motion(joints)

        joint = self.joint(joints).mean(dim=3).permute(0, 2, 1)
        bone = self.bone(bones).mean(dim=3).permute(0, 2, 1)
        motion = self.motion(motion).mean(dim=3).permute(0, 2, 1)

        fused = torch.cat([joint, bone, motion], dim=2)
        x = self.fuse(fused)

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

        for block in self.blocks:
            x = block(x, pad_mask)

        return x, pad_mask

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        seq, pad_mask = self.encode_sequence(x, lengths)
        pooled = self.pool(seq, pad_mask)
        return self.norm(pooled)


class MultiStreamClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        max_len: int,
        gcn_hidden: int = 64,
        gcn_out: int = 96,
        d_model: int = 192,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        conformer_ff: int = 256,
        dropout: float = 0.2,
        proj_dim: int = 128,
        use_cosine: bool = False,
        cosine_scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.encoder = MultiStreamEncoder(
            max_len=max_len,
            gcn_hidden=gcn_hidden,
            gcn_out=gcn_out,
            d_model=d_model,
            conformer_layers=conformer_layers,
            conformer_heads=conformer_heads,
            conformer_ff=conformer_ff,
            dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )
        if use_cosine:
            self.classifier = CosineClassifier(d_model, num_classes, scale=cosine_scale)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor],
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooled = self.encoder(x, lengths)
        logits = self.classifier(pooled)
        if return_embedding:
            return logits, self.projector(pooled)
        return logits, None


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * x @ w.t()
