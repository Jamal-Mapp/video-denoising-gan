import os
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoDenoiseDataset(Dataset):
    def __init__(self, clean_dir, transform=None, noise_type="gaussian", seed=42):
        self.clean_dir = clean_dir
        self.transform = transform
        self.noise_type = noise_type

        self.filenames = sorted(os.listdir(clean_dir))
        random.seed(seed)

    def __len__(self):
        return len(self.filenames)

    def add_noise(self, image):
        image_np = np.array(image).astype(np.float32) / 255.0

        if self.noise_type == "gaussian":
            noise = np.random.normal(0, 0.1, image_np.shape)
        elif self.noise_type == "poisson":
            noise = np.random.poisson(image_np * 255) / 255.0 - image_np
        else:
            raise ValueError("Unsupported noise type")

        noisy_np = np.clip(image_np + noise, 0, 1)
        noisy_img = Image.fromarray((noisy_np * 255).astype(np.uint8))
        return noisy_img

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.filenames[idx])
        clean_img = Image.open(clean_path).convert("RGB")

        noisy_img = self.add_noise(clean_img)

        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, clean_img

