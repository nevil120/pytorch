import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images.
        d_model (int): Size of embedding to turn image into.
        patch_size (int): Size of a single patch.
    """

    def __init__(self,
                 in_channels: int,
                 d_model: int,
                 patch_size: int):

        super().__init__()

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=d_model,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

    def forward(self, x):

        # (batch_size= 32, color_channels=3, height=112, width=112) -->
        # (batch_size= 32, out_channels/d_model=768, no_of_horizontal_patches=14, no_of_vertical_patches=14)
        x_patched = self.patcher(x)

        # (batch_size= 32, out_channels/d_model=768, no_of_horizontal_patches=14, no_of_vertical_patches=14) -->
        # (batch_size= 32, out_channels/d_model=768, no_of_patches=196)
        x_flattened = self.flatten(x_patched)

        # (batch_size= 32, out_channels/d_model=768, flattened_size=196) -->
        # batch_size= 32, no_of_patches=196, out_channels/d_model=768)
        return x_flattened.permute(0, 2, 1)
