"""Fixed-Mel small Conformer epoch encoder; absolute-position project variant."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class FeedForward(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, width=192, heads=4, ff_dim=768, kernel=31, dropout=0.2):
        super().__init__()
        self.ff1 = FeedForward(width, ff_dim, dropout)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution_norm = nn.LayerNorm(width)
        self.convolution = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1), nn.GLU(dim=1),
            nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width),
            nn.BatchNorm1d(width), nn.SiLU(), nn.Conv1d(width, width, 1), nn.Dropout(dropout),
        )
        self.ff2 = FeedForward(width, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        normalized = self.attention_norm(x)
        with sdpa_kernel(SDPBackend.MATH):
            attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.attention_dropout(attention)
        x = x + self.convolution(self.convolution_norm(x).transpose(1, 2)).transpose(1, 2)
        return self.final_norm(x + 0.5 * self.ff2(x))


class ConformerEpochModel(nn.Module):
    embedding_dim = 192

    def __init__(self, layers=2, num_heads=4, ff_dim=768, conv_kernel_size=31, dropout=0.2):
        super().__init__()
        if layers < 1 or num_heads < 1 or 192 % num_heads or ff_dim < 1:
            raise ValueError("Invalid Conformer dimensions.")
        if conv_kernel_size < 1 or conv_kernel_size % 2 != 1 or not 0 <= dropout < 1:
            raise ValueError("Convolution kernel must be positive and odd; dropout must be in [0,1).")
        modules = []
        for source, target in ((1, 32), (32, 64), (64, 96)):
            modules.extend((nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(target), nn.SiLU()))
        self.subsampling = nn.Sequential(*modules)
        self.projection = nn.Linear(96 * 6, 192)
        angles = torch.arange(188).float().unsqueeze(1) * torch.exp(
            torch.arange(0, 192, 2).float() * (-math.log(10000.) / 192)
        )
        self.register_buffer("positions", torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2))
        self.blocks = nn.ModuleList([
            ConformerBlock(192, num_heads, ff_dim, conv_kernel_size, dropout) for _ in range(layers)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(192, 4))

    def encode(self, mel):
        if mel.ndim != 4 or mel.shape[1:] != (1, 48, 1499):
            raise ValueError("Conformer requires fixed Mel shape [B,1,48,1499].")
        features = self.subsampling(mel)
        if features.shape[1:] != (96, 6, 188):
            raise RuntimeError("Unexpected CNN subsampling geometry.")
        tokens = features.permute(0, 3, 1, 2).flatten(2)
        x = self.projection(tokens) + self.positions
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=1)

    def forward(self, mel):
        embedding = self.encode(mel)
        return self.head(embedding), embedding


def conformer_from_config(config):
    model = config["model"]
    required = {"type", "embedding_dim", "layers", "num_heads", "ff_dim", "conv_kernel_size", "dropout"}
    if set(model) != required or model["type"] != "conformer_epoch" or model["embedding_dim"] != 192:
        raise ValueError("Unknown/missing Conformer model options or embedding dimension.")
    for key in ("layers", "num_heads", "ff_dim", "conv_kernel_size"):
        if type(model[key]) is not int:
            raise ValueError("Conformer dimensions must be integers.")
    return ConformerEpochModel(**{key: model[key] for key in required - {"type", "embedding_dim"}})
