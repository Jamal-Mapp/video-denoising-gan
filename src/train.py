import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data_loader import VideoDenoiseDataset
from model import Generator, Discriminator
import torchvision.transforms as T

# ---------- Load Config ----------
def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------- Training Function ----------
def train():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    # Create output folder
    os.makedirs("results", exist_ok=True)

    # ---------- Data ----------
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    dataset = VideoDenoiseDataset(
        clean_dir=config["dataset_path"],
        transform=transform,
        noise_type="gaussian"
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # ---------- Models ----------
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    # ---------- Losses ----------
    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    # ---------- Optimizers ----------
    g_opt = optim.Adam(gen.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
    d_opt = optim.Adam(disc.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))

    for epoch in range(config["epochs"]):
        for i, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # ---------------- Train Discriminator ----------------
            fake = gen(noisy)
            real_pred = disc(clean)
            fake_pred = disc(fake.detach())

            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)

            d_loss_real = bce(real_pred, real_labels)
            d_loss_fake = bce(fake_pred, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            disc.zero_grad()
            d_loss.backward()
            d_opt.step()

            # ---------------- Train Generator ----------------
            fake_pred = disc(fake)
            adv_loss = bce(fake_pred, real_labels)
            recon_loss = l1(fake, clean)
            g_loss = adv_loss + 100 * recon_loss  # Weighted sum

            gen.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i % 10 == 0:
                print(f"🌀 Epoch [{epoch+1}/{config['epochs']}], Step [{i+1}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # ---------- Save sample images ----------
        save_image(torch.cat([noisy[:4], fake[:4], clean[:4]], dim=0),
                   f"results/epoch_{epoch+1:03d}.png", nrow=4, normalize=True)

if __name__ == "__main__":
    train()
