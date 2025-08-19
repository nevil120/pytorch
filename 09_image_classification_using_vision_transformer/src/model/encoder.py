import torch.nn as nn

from src.model.encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Encoder containing multiple EncoderBlocks
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.layers_list = []

        # Create n(=num_layers) EncoderBlocks
        for _ in range(num_layers):
            self.layers_list.append(EncoderBlock(d_model, num_heads, d_ff, dropout))

        self.layers = nn.ModuleList(self.layers_list)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
