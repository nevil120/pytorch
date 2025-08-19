import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding creates positional embeddings and returns
    positional embeddings added to a input embeddings
    """

    def __init__(self, d_model: int, no_of_patches: int, dropout: float):
        super().__init__()
        self.embedding_dim = d_model
        self.no_of_patches = no_of_patches
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix (no_of_patches * d_model)
        positional_encoding = torch.zeros(no_of_patches, d_model)

        numerator_term = torch.arange(0, no_of_patches, dtype=torch.float).unsqueeze(dim=1)
        denominator_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply a function to a positional encoding matrix
        positional_encoding[:, 0::2] = torch.sin(numerator_term * denominator_term)
        positional_encoding[:, 1::2] = torch.cos(numerator_term * denominator_term)

        # Add another dimension to accommodate batch of sentences - (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # Adds input embedding to a positional embeddings,
        # (batch_size, no_of_patches, embedding_dim) + (1, no_of_patches, embedding_dim)
        # Outputs positional encoded embeddings (batch_size, no_of_patches, embedding_dim)
        x = x + (self.positional_encoding[:, :, :]).requires_grad_(False)
        return self.dropout(x)
