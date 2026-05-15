import torch
import torch.nn as nn
 
class Encoder(nn.Module):
    """
    Encoder compresses the input image.
 
    Each layer halves dims and 2x channel depth using strided convolutions (stride=2).
 
    Layer dims:
    input -> (B, 3,   256, 256)
       #1 -> (B, 32,  128, 128)
       #2 -> (B, 64,   64,  64)
       #3 -> (B, 128,  32,  32)
       #4 -> (B, 256,  16,  16)
       #5 -> (B, 512,   8,   8)
    """
 
    def __init__(self):
        super().__init__()
 
        self.layers = nn.Sequential(
            #1
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #3
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #5
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
 
    def forward(self, x):
        return self.layers(x)