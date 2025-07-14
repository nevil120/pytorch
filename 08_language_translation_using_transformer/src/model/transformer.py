import torch
import torch.nn as nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.input_embedding import InputEmbedding
from src.model.positional_encoding import PositionalEncoding
from src.model.projection_layer import ProjectionLayer


class Transformer(nn.Module):

    def __init__(self, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding,
                 encoder: Encoder, decoder: Decoder, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(d_model, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      dropout: float, num_layers: int, num_heads: int, d_ff: int):

    src_embeddings = InputEmbedding(d_model, src_vocab_size)
    tgt_embeddings = InputEmbedding(d_model, tgt_vocab_size)

    src_positional_encodings = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encodings = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(src_embeddings, tgt_embeddings, src_positional_encodings, tgt_positional_encodings,
                              encoder, decoder, proj_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
