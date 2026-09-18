"""Small, non-causal temporal Transformer over frozen 30-second embeddings."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .windows import context_options


class TemporalTransformer(nn.Module):
    def __init__(self, d_model=192, layers=2, num_heads=4, ff_dim=512,
                 dropout=0.2, input_epochs=40):
        super().__init__()
        context_options({"input_epochs": input_epochs})
        if d_model != 192 or layers < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("Expected d_model=192 and positive compatible layers/heads.")
        self.input_epochs = input_epochs
        self.left_context = (input_epochs - 20) // 2
        self.input_norm = nn.LayerNorm(d_model)
        # Construct independently: cloned identical initial layers are not intended.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward=ff_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 4))
        frequencies = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        self.register_buffer("position_frequencies", frequencies)

    def forward_all(self, embeddings, input_valid, epoch_indices):
        """Return [B,L,4]; invalid rows/queries are zero, never all-masked attention."""
        if embeddings.ndim != 3 or embeddings.shape[-1] != 192:
            raise ValueError("Expected embedding shape [B,L,192].")
        if input_valid.shape != embeddings.shape[:2] or input_valid.dtype != torch.bool:
            raise ValueError("Expected boolean input_valid [B,L].")
        if epoch_indices.shape != input_valid.shape or epoch_indices.dtype != torch.long:
            raise ValueError("Expected int64 epoch_indices [B,L].")
        if ((epoch_indices < 0) & input_valid).any():
            raise ValueError("Valid input must have a real nonnegative epoch index.")
        nonempty = input_valid.any(dim=1)
        result = embeddings.new_zeros((*input_valid.shape, 4), dtype=torch.float32)
        if not nonempty.any():
            return result
        valid = input_valid[nonempty]
        x = embeddings[nonempty].masked_fill(~valid.unsqueeze(-1), 0.)
        positions = epoch_indices[nonempty].masked_fill(~valid, 0).float()
        angles = positions.unsqueeze(-1) * self.position_frequencies
        positional = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        x = (self.input_norm(x) + positional).masked_fill(~valid.unsqueeze(-1), 0.)
        # Short windows: math SDPA keeps deterministic backward without a new dependency.
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=~valid)
        logits = self.head(self.final_norm(x)).float()
        logits = logits.masked_fill(~valid.unsqueeze(-1), 0.)
        return result.index_copy(0, nonempty.nonzero().flatten(), logits)

    def forward(self, embeddings, input_valid, epoch_indices):
        if embeddings.ndim != 3 or embeddings.shape[1] != self.input_epochs:
            raise ValueError("Embedding length does not match configured context.")
        logits = self.forward_all(embeddings, input_valid, epoch_indices)
        return logits[:, self.left_context:self.left_context + 20]


def transformer_from_config(config):
    model = config["model"]
    return TemporalTransformer(
        d_model=model["d_model"], layers=model["layers"],
        num_heads=model["num_heads"], ff_dim=model["ff_dim"],
        dropout=model["dropout"], input_epochs=config["data"]["input_epochs"],
    )
