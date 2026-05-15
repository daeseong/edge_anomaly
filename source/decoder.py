import torch
import torch.nn as nn
 
import torch.nn as nn
 
 
class Decoder(nn.Module):
    """
    Decode reconstructs the image.
 
    Each layer 2x dims and halves channel depth.
 
    Layer dims:
    input -> (B, 512,   8,   8)
       #1 -> (B, 256,  16,  16)
       #2 -> (B, 128,  32,  32)
       #3 -> (B, 64,   64,  64)
       #4 -> (B, 32,  128, 128)
       #5 -> (B, 3,   256, 256)
    """
 
    def __init__(self):
        super().__init__()
 
        self.layers = nn.Sequential(
            #1
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #3
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #5
            # Sigmoid bounds out to [0, 1] to match normalized image range
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
 
    def forward(self, x):
        return self.layers(x)
 