import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import random
import os

from cvae_model import CVAE

class WavLMDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.speaker_to_id = {}
        self.root_dir = Path(root_dir)
        for i, speaker_dir in enumerate(sorted(self.root_dir.iterdir())):
            self.speaker_to_id[speaker_dir.name] = i
            feats = torch.load(speaker_dir / f"{speaker_dir.name}.pt")  # shape (N, 1024)
            for vec in feats:
                self.samples.append((vec, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, speaker_id = self.samples[idx]
        return vec, speaker_id

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl * 0.0001

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, speaker_ids in tqdm(dataloader):
        x = x.to(device).float()
        speaker_ids = speaker_ids.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, speaker_ids)
        loss = loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main(data_dir, epochs=10, batch_size=512, save_path="cvae_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WavLMDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = CVAE(num_speakers=len(dataset.speaker_to_id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/train")  # Path to training data
