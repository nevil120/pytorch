import math

import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    InputEmbeddings creates and returns embeddings for the input sentence.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        :param d_model: Size of embedding vector
        :param vocab_size: Vocabulary size of the input words
        :return: returns nothing
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
