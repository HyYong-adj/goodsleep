from __future__ import annotations

import torch
from torch import nn
from .windows import context_options


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
    def __init__(self, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.2, input_epochs: int = 40, target_epochs: int = 20, time_feature: bool = False):
        super().__init__()
        context_options(dict(input_epochs=input_epochs, target_epochs=target_epochs))
        self.input_epochs, self.target_epochs = input_epochs, target_epochs
        self.left_context = (input_epochs - target_epochs) // 2
        self.norm = nn.LayerNorm(192)
        # A1: elapsed-time prior. [h, h^2] projected onto the embedding, matching the
        # night-macrostructure cue (N3 early, REM late) that a purely acoustic
        # embedding cannot express. Absent by default so existing checkpoints load.
        self.time_proj = nn.Linear(2, 192) if time_feature else None
        self.lstm = nn.LSTM(192, hidden_dim, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 4))

    def forward(self, embeddings: torch.Tensor, input_valid: torch.Tensor | None = None, hours: torch.Tensor | None = None) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1:] != (self.input_epochs, 192):
            raise ValueError("Embedding input shape does not match configured context.")
        if input_valid is not None:
            embeddings = embeddings.masked_fill(~input_valid.unsqueeze(-1), 0.0)
        if self.time_proj is not None:
            if hours is None:
                raise ValueError("time_feature=True requires the hours tensor.")
            if hours.shape != embeddings.shape[:2]:
                raise ValueError("hours must be [batch, input_epochs].")
            h = hours.to(embeddings.dtype).unsqueeze(-1)
            embeddings = embeddings + self.time_proj(torch.cat([h, h * h], dim=-1))
        output, _ = self.lstm(self.norm(embeddings))
        return self.head(output[:, self.left_context:self.left_context + self.target_epochs])


def window_hours(target_indexes: torch.Tensor, input_valid: torch.Tensor, left_context: int, hours_scale: float = 8.0) -> torch.Tensor:
    """Elapsed recording hours for each input position, zeroed on padding.

    ``target_indexes[:, 0]`` is the window's first target epoch on the real
    30-second recording grid, so input position j sits at
    ``target_start - left_context + j``. Padded positions get 0, as in the
    reference implementation, rather than a negative or out-of-range time.
    """
    start = target_indexes[:, 0].to(torch.float32)
    offsets = torch.arange(input_valid.shape[1], dtype=torch.float32, device=start.device)
    index = start.unsqueeze(1) - float(left_context) + offsets.unsqueeze(0)
    hours = index * 30.0 / 3600.0 / float(hours_scale)
    return hours.masked_fill(~input_valid.to(hours.device), 0.0)
