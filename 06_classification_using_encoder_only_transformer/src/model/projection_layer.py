import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Final projection layer to project the output to number of classes
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.linear_layer = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, 1, d_model) --> (batch, 1, num_classes)
        return self.linear_layer(torch.mean(x, -2))
