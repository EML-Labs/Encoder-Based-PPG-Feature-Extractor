import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleDecoder(nn.Module):
    def __init__(self,latent_size=32):
        super(SimpleDecoder,self).__init__()
        self.fc = nn.Linear(latent_size, 64 * 125)  # (latent_size) -> (64*125)
        self.tcnv_01 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, 125) -> (32, 125)
            nn.BatchNorm1d(32)
        )
        self.tcnv_02 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # (32, 125) -> (16, 250)
            nn.BatchNorm1d(16)
        )
        self.tcnv_03 = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),  # (16, 250) -> (1, 250)
            nn.BatchNorm1d(1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 125)  # Reshape to (batch_size, 64, 125)
        x = F.relu(self.tcnv_01(x))
        x = F.relu(self.tcnv_02(x))
        x = torch.tanh(self.tcnv_03(x))  # Use tanh to keep output in [-1, 1]
        return x