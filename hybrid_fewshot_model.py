import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from hybrid_model import GraphBackbone, AttnPool1D


class HybridEncoder(nn.Module):
    def __init__(
        self,
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

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return self.projector(emb)


class ProtoNet(nn.Module):
    def __init__(self, encoder: nn.Module, use_cosine: bool = True, init_logit_scale: float = 10.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_cosine = use_cosine
        if use_cosine:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale)))
        else:
            self.logit_scale = None

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
        emb_query = self.encoder(query_x, query_len)

        if self.use_cosine:
            emb_support = F.normalize(emb_support, dim=1)
            prototypes = emb_support.view(n_way, k_shot, -1).mean(dim=1)
            prototypes = F.normalize(prototypes, dim=1)
            emb_query = F.normalize(emb_query, dim=1)
            logits = emb_query @ prototypes.T
            return logits * self.logit_scale.exp()

        prototypes = emb_support.view(n_way, k_shot, -1).mean(dim=1)
        dists = torch.cdist(emb_query, prototypes)
        return -dists
