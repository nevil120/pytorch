import torch.nn as nn

from src.model.feed_forward import FeedForward
from src.model.multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """
    EncoderBlock containing self-attention and feed forward layers.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-Attention with Residual Connection
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm(x)

        # FeedForward with Residual Connection
        x = x + self.dropout(self.feed_forward(x))
        x = self.norm(x)

        return x
