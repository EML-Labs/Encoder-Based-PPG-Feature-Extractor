
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self,latent_size=32,dropout_rate=0.1):
        super(SimpleEncoder,self).__init__()
        self.cnv_01 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),  # (1, 250) -> (16, 250)
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.cnv_02 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),  # (16, 250) -> (32, 125)
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.cnv_03 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),  # (32, 125) -> (64, 125)
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc = nn.Linear(64 * 125, latent_size)  # (64*125) -> (latent_size)


    def forward(self, x):
        x = self.cnv_01(x)
        x = self.cnv_02(x)
        x = self.cnv_03(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    