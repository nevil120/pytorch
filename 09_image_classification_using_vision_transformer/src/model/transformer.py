import torch
import torch.nn as nn

from src.model.encoder import Encoder
from src.model.patch_embedding import PatchEmbedding
from src.model.positional_encoding import PositionalEncoding
from src.model.projection_layer import ProjectionLayer


class Transformer(nn.Module):
    """
    Transformer using PatchEmbedding, PositionalEncoding, Encoder, and Projection Layer
    """

    def __init__(self, d_model: int, embed: PatchEmbedding, pos: PositionalEncoding,
                 encoder: Encoder, projection_layer: ProjectionLayer):
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.embed = embed
        self.pos = pos
        self.encoder = encoder
        self.projection_layer = projection_layer

    def encode(self, input):
        # (batch, color_channel, height, width) --> (batch, no_of_patches, d_model)
        input = self.embed(input)
        # (batch_size, 1, embed_dim)
        class_tokens = self.class_token.expand(input.size(0), -1, -1)
        # (batch_size, 1, embed_dim) + (batch_size, num_patches, embed_dim) --> (batch_size, num_patches+1, embed_dim)
        input = torch.cat((class_tokens, input), dim=1)
        # (batch, no_of_patches+1, d_model) --> (batch, no_of_patches+1, d_model)
        input = self.pos(input)
        # (batch, no_of_patches+1, d_model) --> (batch, no_of_patches+1, d_model)
        return self.encoder(input)

    def project(self, x):
        # (batch, no_of_patches, d_model) --> (batch, num_classes)
        return self.projection_layer(x)


def build_transformer(in_channels: int, d_model: int, patch_size: int, no_of_patches: int, dropout: float,
                      num_layers: int, num_heads: int, d_ff: int, num_classes: int):

    patch_embeddings = PatchEmbedding(in_channels, d_model, patch_size)

    patch_positional_encoded_embeddings = PositionalEncoding(d_model, no_of_patches, dropout)

    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)

    proj_layer = ProjectionLayer(d_model, num_classes)

    transformer = Transformer(d_model, patch_embeddings, patch_positional_encoded_embeddings, encoder, proj_layer)

    # Initialize the model parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
