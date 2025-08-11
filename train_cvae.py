import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

from cvae_model import CVAE

class WavLMDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.speaker_dirs = sorted(self.root_dir.iterdir())
        self.speaker_to_id = {d.name: i for i, d in enumerate(self.speaker_dirs)}

        # Count number of features per speaker (lightweight)
        self.lengths = []
        for speaker_dir in tqdm(self.speaker_dirs, desc="Counting features per speaker"):
            feats = torch.load(speaker_dir / f"{speaker_dir.name}.pt", map_location='cpu')
            self.lengths.append(len(feats))

        # Build cumulative sum for global indexing
        self.cumsum = [0] + list(torch.cumsum(torch.tensor(self.lengths), dim=0).numpy())

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        # Find speaker index via cumsum
        speaker_idx = next(i for i in range(len(self.lengths)) if self.cumsum[i+1] > idx)
        sample_idx = idx - self.cumsum[speaker_idx]

        speaker_dir = self.speaker_dirs[speaker_idx]
        feats = torch.load(speaker_dir / f"{speaker_dir.name}.pt", map_location='cpu')

        vector = feats[sample_idx]
        speaker_id = speaker_idx
        return vector, speaker_id

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

def main(data_dir, epochs=10, batch_size=128, save_path="cvae_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WavLMDataset(data_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    model = CVAE(num_speakers=len(dataset.speaker_to_id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/train_mini")
