import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Final projection layer to project the output to number of classes
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.linear_layer(x)
