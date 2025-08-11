import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np

from cvae_model import CVAE

class LazySpeakerFeatureDataset(Dataset):
    def __init__(self, root_dir, features_per_speaker=8996):
        self.root_dir = Path(root_dir)
        self.features_per_speaker = features_per_speaker
        
        print("Loading speaker directories...")
        self.speaker_dirs = []
        for p in tqdm(sorted([p for p in self.root_dir.iterdir() if p.is_dir()]), desc="Speakers"):
            self.speaker_dirs.append(p)
        self.speaker_names = [p.name for p in self.speaker_dirs]
        self.total_samples = len(self.speaker_dirs) * self.features_per_speaker

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        speaker_idx = idx // self.features_per_speaker
        feature_idx = idx % self.features_per_speaker

        speaker_name = self.speaker_names[speaker_idx]
        feature_path = self.root_dir / speaker_name / f"{speaker_name}.npy"
        
        # Load .npy file every time (lazy loading, avoids RAM bloat)
        features = np.load(feature_path)
        
        # Convert single feature vector to torch tensor
        feature_vec = torch.from_numpy(features[feature_idx]).float()
        speaker_id = speaker_idx
        return feature_vec, speaker_id

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl * 0.0001

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, speaker_ids in tqdm(dataloader, desc="Training batches"):
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
    print("Loading training data...")
    dataset = LazySpeakerFeatureDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    model = CVAE(num_speakers=len(dataset.speaker_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loss = train(model, dataloader, optimizer, device)
        print(f"Loss: {loss:.4f}")
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/train_mini")
