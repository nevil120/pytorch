import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Final projection layer to project the output to output vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, tgt_seq_len, d_model) --> (batch, tgt_seq_len, tgt_vocab_size)
        # return torch.log_softmax(self.linear_layer(x), dim=-1)
        return self.linear_layer(x)
