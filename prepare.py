from Dataset import ContrastivePPGDataset
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Utils import get_logger

from Models.AutoEncoders import SimpleAutoencoder

logger = get_logger()

data_folder = os.path.join(os.getcwd(), 'Data')
dataset = ContrastivePPGDataset(folder_path=data_folder, sampling_rate=125, window_size=250, quality_threshold=0.90, padding=1, shift=25, num_pairs_per_class=1000)

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
# Freeze decoder
for param in denosing_autoencoder.decoder.parameters():
    param.requires_grad = False
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denosing_autoencoder.to(device)


logger.info(f"Model and datasets moved to device: {device}")

criterion = nn.MSELoss()
optimizer = optim.Adam(denosing_autoencoder.encoder.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

num_epochs = 1

for epoch in range(num_epochs):
    denosing_autoencoder.train()
    train_loss = 0.0
    for s1, s2, label in train_dataloader:
        s1 = s1.to(device)  # Move inputs to GPU
        s2 = s2.to(device)  # Move targets to GPU
        label = label.to(device)  # Move labels to GPU

        optimizer.zero_grad()
        z1 = denosing_autoencoder.encoder(s1)
        z2 = denosing_autoencoder.encoder(s2)
        cos_sim = nn.functional.cosine_similarity(z1, z2)
        loss = criterion(cos_sim, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * label.size(0)
    train_loss /= len(train_dataloader.dataset)

    denosing_autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for s1, s2, label in val_dataloader:
            s1 = s1.to(device)  # Move inputs to GPU
            s2 = s2.to(device)  # Move targets to GPU
            label = label.to(device)  # Move labels to GPU
            z1 = denosing_autoencoder.encoder(s1)
            z2 = denosing_autoencoder.encoder(s2)
            cos_sim = nn.functional.cosine_similarity(z1, z2)
            loss = criterion(cos_sim, label)
            val_loss += loss.item() * label.size(0)
    val_loss /= len(val_dataloader.dataset)

    scheduler.step(val_loss)  # Reduce LR on plateau



    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')