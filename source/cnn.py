import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
 
 
class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for unsupervised anomaly detection.
 
    Model learns to reconstruct normal images well, but fails on unseen defects.
 
    Input:  (B, 3, 256, 256)
    Output: (B, 3, 256, 256)
    """
 
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dequant = torch.quantization.DeQuantStub()
 
    def forward(self, x):
        x = self.quant(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.dequant(x)
        return x
 
 
if __name__ == "__main__":
    # Verify that input dims = output dims
    model = ConvAutoencoder()
    d_in = torch.randn(1, 3, 256, 256)
    d_out = model(d_in)
 
    print(f"Input shape: {d_in.shape}")
    print(f"Output shape: {d_out.shape}")
    assert d_in.shape == d_out.shape, "INPUT AND OUTPUT DIMS DO NOT MATCH"
    print("Shape matches, PASSED!")