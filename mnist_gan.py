import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configurations ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 0.0002
BETA1 = 0.5
LATENT_DIM = 100
IMG_SIZE = 784  # 28x28
EPOCHS = 100
SAMPLE_INTERVAL = 2
SAVE_DIR = "samples"
MODEL_DIR = "models"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- Dataset ---
class MNISTCSVDataset(Dataset):
    def __init__(self, csv_file):
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # In this specific file, there are 784 columns (pixel0 to pixel783)
        # So we just take all columns.
        self.pixels = df.values
            
        self.pixels = (self.pixels.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return torch.tensor(self.pixels[idx], dtype=torch.float32)

# --- Models ---
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# --- Training ---
def train():
    # Load data
    print("Loading dataset...")
    dataset = MNISTCSVDataset('mnist_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models
    generator = Generator(LATENT_DIM, IMG_SIZE).to(DEVICE)
    discriminator = Discriminator(IMG_SIZE).to(DEVICE)

    # Loss and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)

            # Labels
            valid = torch.ones(batch_size, 1, device=DEVICE)
            fake = torch.zeros(batch_size, 1, device=DEVICE)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        if (epoch + 1) % SAMPLE_INTERVAL == 0:
            save_samples(generator, epoch + 1)
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f"generator_epoch_{epoch+1}.pth"))
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, "generator_latest.pth"))

    print("Training complete!")

def save_samples(generator, epoch):
    z = torch.randn(25, LATENT_DIM, device=DEVICE)
    gen_imgs = generator(z).detach().cpu().numpy()
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]
    
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    
    path = os.path.join(SAVE_DIR, f"epoch_{epoch}.png")
    fig.savefig(path)
    plt.close()
    print(f"Saved samples to {path}")

if __name__ == "__main__":
    train()
