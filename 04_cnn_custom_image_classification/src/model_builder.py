import torch
from torch import nn

class ImageClassificationModel(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, output_channels):
        
        super().__init__()
        
        # Feature extraction layer 1
        self.layer_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Feature extraction layer 2
        self.layer_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Classification layer
        self.layer_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels*16*16, output_channels)
        )
     
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
