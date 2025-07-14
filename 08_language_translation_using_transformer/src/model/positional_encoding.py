import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding creates positional embeddings and returns
    positional embeddings added to a input embeddings
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        :param d_model: Size of embedding vector
        :param seq_len: Size of a sentence
        :param dropout: Dropout rate
        :return: returns nothing
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix (seq_len * d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        numerator_term = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=1)
        denominator_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply a function to a positional encoding matrix
        positional_encoding[:, 0::2] = torch.sin(numerator_term * denominator_term)
        positional_encoding[:, 1::2] = torch.cos(numerator_term * denominator_term)

        # Add another dimension to accommodate batch of sentences - (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # Add input embedding to a positional embeddings - (batch_size, seq_len, d_model) + (1, seq_len, d_model)
        # Outputs (batch_size, seq_len, d_model)
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
