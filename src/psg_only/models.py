from __future__ import annotations

import torch
from torch import nn


class DepthwiseBlock(nn.Module):
    def __init__(self, source: int, target: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(source, source, 3, stride, 1, groups=source, bias=False),
            nn.BatchNorm2d(source),
            nn.ReLU6(),
            nn.Conv2d(source, target, 1, bias=False),
            nn.BatchNorm2d(target),
            nn.ReLU6(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AcousticEncoder(nn.Module):
    embedding_dim = 192

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            DepthwiseBlock(32, 64, 2),
            DepthwiseBlock(64, 96, 2),
            DepthwiseBlock(96, 128, 2),
            DepthwiseBlock(128, 160, 2),
            nn.Conv2d(160, self.embedding_dim, 1, bias=False),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class B0Model(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.encoder = AcousticEncoder()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(192, 4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return self.head(embedding), embedding


class B1Model(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(192)
        self.lstm = nn.LSTM(192, hidden_dim, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 4))

    def forward(self, embeddings: torch.Tensor, input_valid: torch.Tensor | None = None) -> torch.Tensor:
        if input_valid is not None:
            embeddings = embeddings.masked_fill(~input_valid.unsqueeze(-1), 0.0)
        output, _ = self.lstm(self.norm(embeddings))
        return self.head(output[:, 10:30])
