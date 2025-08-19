import torch.nn as nn

from src.model.embedding import Embedding
from src.model.positional_encoding import PositionalEncoding
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.projection_layer import ProjectionLayer


class Transformer(nn.Module):
    """
    Transformer using InputEmbedding, PositionalEncoding, Encoder, and Projection Layer
    """

    def __init__(self, src_embed: Embedding, tgt_embed: Embedding, src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        # (batch, src_seq_len) --> (batch, src_seq_len, d_model)
        x = self.src_embed(x)
        # (batch, src_seq_len, d_model) --> (batch, src_seq_len, d_model)
        x = self.src_pos(x)
        # (batch, src_seq_len, d_model) --> (batch, src_seq_len, d_model)
        return self.encoder(x, src_mask)

    def decode(self, y, encoder_output, src_mask):
        # (batch, tgt_seq_len) --> (batch, tgt_seq_len, d_model)
        y = self.tgt_embed(y)
        # (batch, tgt_seq_len, d_model) --> (batch, tgt_seq_len, d_model)
        y = self.tgt_pos(y)
        # (batch, tgt_seq_len, d_model) --> (batch, tgt_seq_len, d_model)
        return self.decoder(y, encoder_output, src_mask)

    def project(self, decoder_output):
        # (batch, tgt_seq_len, d_model) --> (batch, tgt_seq_len, tgt_vocab_size)
        return self.projection_layer(decoder_output)


def build_transformer(d_model, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      dropout: float, num_layers: int, num_heads: int, d_ff: int):

    input_embeddings = Embedding(d_model, src_vocab_size)
    output_embeddings = Embedding(d_model, tgt_vocab_size)

    input_positional_encoded_embeddings = PositionalEncoding(d_model, src_seq_len, dropout)
    output_positional_encoded_embeddings = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(input_embeddings, output_embeddings, input_positional_encoded_embeddings,
                              output_positional_encoded_embeddings, encoder, decoder, proj_layer)

    # Initialize the model parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
