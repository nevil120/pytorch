import math

import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    InputEmbedding creates and returns embeddings for the input sentences.
    """

    # Creates initial tensor of size (vocab_size, d_model)
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
