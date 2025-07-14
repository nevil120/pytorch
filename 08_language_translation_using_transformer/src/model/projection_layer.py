import torch.nn as nn


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear_layer(x)
