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

denosing_autoencoder = SimpleAutoencoder(latent_size=32)
logger.info("Denoising Autoencoder model initialized.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denosing_autoencoder.to(device)


logger.info(f"Model and datasets moved to device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(denosing_autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

num_epochs = 20

for epoch in range(num_epochs):
    denosing_autoencoder.train()
    train_loss = 0.0
    for x,y in train_dataloader:
        x = x.to(device)  # Move inputs to GPU
        y = y.to(device)  # Move targets to GPU
        optimizer.zero_grad()
        outputs = denosing_autoencoder(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y.size(0)
    train_loss /= len(train_dataloader.dataset)

    denosing_autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x,y in val_dataloader:
            x = x.to(device)  # Move inputs to GPU
            y = y.to(device)  # Move targets to GPU
            outputs = denosing_autoencoder(x)
            loss = criterion(outputs, y)
            val_loss += loss.item() * y.size(0)
    val_loss /= len(val_dataloader.dataset)

    scheduler.step(val_loss)  # Reduce LR on plateau



    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')