import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention calculates attention among patches in a image to
    find out how patches are related to each other.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert(d_model % num_heads == 0), "d_model is not divisible by number of heads"
        self.d_k = d_model // num_heads

        self.query_linear_layer = nn.Linear(d_model, d_model)
        self.key_linear_layer = nn.Linear(d_model, d_model)
        self.value_linear_layer = nn.Linear(d_model, d_model)

        self.output_linear_layer = nn.Linear(d_model, d_model)

    # Calculate attention for the input batch of sentences
    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, num_heads, no_of_patches, d_k) @ (batch, num_heads, d_k, no_of_patches)
        # --> (batch, num_heads, no_of_patches, no_of_patches)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (attention_scores @ value)
        # (batch, num_heads, no_of_patches, no_of_patches) @ (batch, num_heads, no_of_patches, d_k)
        # --> (batch, num_heads, no_of_patches, d_k)
        return attention_scores @ value

    def forward(self, query_input, key_input, value_input):
        # Applying linear layer to input matrices to create query, key, and values
        # (batch, no_of_patches, d_model) --> (batch, no_of_patches, d_model)
        query = self.query_linear_layer(query_input)
        key = self.key_linear_layer(key_input)
        value = self.value_linear_layer(value_input)

        # Divide it into number of attention heads
        # (batch, no_of_patches, d_model) --> (batch, no_of_patches, num_heads, d_k)
        # --> (batch, num_heads, no_of_patches, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Calculate attention for the input batch of sentences
        x = MultiHeadAttention.attention(query, key, value, self.dropout)

        # Combine attention heads
        # (batch, num_heads, seq_len, d_k) --> (batch, seq_len, num_heads, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # Applying linear layer to final output
        return self.output_linear_layer(x)
