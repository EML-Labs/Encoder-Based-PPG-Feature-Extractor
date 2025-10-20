from Dataset import PPGDataset
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Utils import get_logger

from Models.AutoEncoders import SimpleAutoencoder

logger = get_logger()

data_folder = os.path.join(os.getcwd(), 'Data', 'SR-Data')
dataset = PPGDataset(folder_path=data_folder, sampling_rate=125, window_size=250, quality_threshold=0.99, padding=1, shift=1)

train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False,pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

logger.info(f"Dataset split into Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

denosing_autoencoder = SimpleAutoencoder(latent_size=32, dropout_rate=0.1)
logger.info("Denoising Autoencoder model initialized.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denosing_autoencoder.to(device)
train_dataset = train_dataset.to(device)
val_dataset = val_dataset.to(device)

logger.info(f"Model and datasets moved to device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(denosing_autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)