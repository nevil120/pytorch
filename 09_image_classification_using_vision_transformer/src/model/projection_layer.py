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
        # (batch, no_of_patches+1, d_model) --> (batch, 1, d_model) --> (batch, num_classes)
        # Put 1st patch (class token) logits through projection
        return self.linear_layer(x[:, 0])
