import torch.nn as nn
from Models.Encoders import SimpleEncoder
from Models.Decoders import SimpleDecoder

class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_size=32, dropout_rate=0.1):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = SimpleEncoder(latent_size, dropout_rate)
        self.decoder = SimpleDecoder(latent_size, dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x