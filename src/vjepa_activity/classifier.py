"""Pooled MLP used by the notebook's V-JEPA2 classifier."""

import torch.nn as nn


class PooledMLPClassifier(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.mlp(features)
