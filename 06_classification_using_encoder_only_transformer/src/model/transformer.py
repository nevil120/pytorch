import torch.nn as nn

from src.model.encoder import Encoder
from src.model.input_embedding import InputEmbedding
from src.model.positional_encoding import PositionalEncoding
from src.model.projection_layer import ProjectionLayer


class Transformer(nn.Module):
    """
    Transformer using InputEmbedding, PositionalEncoding, Encoder, and Projection Layer
    """

    def __init__(self, embed: InputEmbedding, pos: PositionalEncoding,
                 encoder: Encoder, projection_layer: ProjectionLayer):
        super().__init__()
        self.embed = embed
        self.pos = pos
        self.encoder = encoder
        self.projection_layer = projection_layer

    def encode(self, input, mask):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        input = self.embed(input)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        input = self.pos(input)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.encoder(input, mask)

    def project(self, x):
        # (batch, seq_len, d_model) --> (batch, 1, num_classes)
        return self.projection_layer(x)


def build_transformer(d_model, src_vocab_size: int, src_seq_len: int, dropout: float,
                      num_layers: int, num_heads: int, d_ff: int, num_classes: int):

    input_embeddings = InputEmbedding(d_model, src_vocab_size)

    input_positional_encoded_embeddings = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)

    proj_layer = ProjectionLayer(d_model, num_classes)

    transformer = Transformer(input_embeddings, input_positional_encoded_embeddings, encoder, proj_layer)

    # Initialize the model parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
