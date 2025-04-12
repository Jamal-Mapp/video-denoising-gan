import torch
from model import Generator, Discriminator

def train():
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    print("✅ Models initialized")

if __name__ == "__main__":
    train()

