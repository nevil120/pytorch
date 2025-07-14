import torch.nn as nn

from src.model.decoder_block import DecoderBlock


class Decoder(nn.Module):

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.layers_list = []

        for _ in range(num_layers):
            self.layers_list.append(DecoderBlock(d_model, num_heads, d_ff, dropout))

        self.layers = nn.ModuleList(self.layers_list)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
