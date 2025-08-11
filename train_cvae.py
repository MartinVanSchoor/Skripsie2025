# train_cvae.py (fault-tolerant, resumable)
import os
import glob
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np

from cvae_model import CVAE

# ---------------------------
# Dataset (lazy .npy loading)
# ---------------------------
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

        # Load .npy file every time (lazy loading)
        features = np.load(feature_path, mmap_mode=None)
        feature_vec = torch.from_numpy(features[feature_idx]).float()
        # free numpy array reference asap (help GC)
        del features
        return feature_vec, speaker_idx

# ---------------------------
# Loss and helpers
# ---------------------------
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl * 0.0001

def save_checkpoint(state, checkpoint_dir, name="cvae_checkpoint.pt"):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / name
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")

def find_latest_checkpoint(checkpoint_dir, pattern="cvae_checkpoint_epoch_*.pt"):
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

# ---------------------------
# Training loop (resumable)
# ---------------------------
def train_epoch(model, dataloader, optimizer, device, start_batch=0,
                checkpoint_dir="checkpoints", checkpoint_every=200,
                empty_cache_every=50):
    model.train()
    total_loss = 0.0
    batch_idx = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training batches", initial=start_batch)
    for i, (x, speaker_ids) in pbar:
        # skip already-processed batches if resuming within epoch
        if i < start_batch:
            continue

        # move to device
        x = x.to(device, non_blocking=True).float()
        speaker_ids = speaker_ids.to(device, non_blocking=True)

        # quick NaN/Inf check
        if not torch.isfinite(x).all():
            print(f"Non-finite values detected in batch {i}; skipping batch.")
            continue

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, speaker_ids)
        loss = loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_idx = i

        # Update progress bar
        pbar.set_postfix(loss=loss.item())

        # periodic cache clear to reduce fragmentation
        if (i + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

        # periodic checkpoint
        if (i + 1) % checkpoint_every == 0:
            ckpt_name = f"cvae_checkpoint_epoch_{train_epoch.current_epoch}_batch_{i+1}.pt"
            state = {
                "epoch": train_epoch.current_epoch,
                "batch": i + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir, ckpt_name)

    return total_loss / max(1, (batch_idx + 1))

# attach attribute for epoch tracking (mutable)
train_epoch.current_epoch = 0

# ---------------------------
# Main entry (resume support)
# ---------------------------
def main(data_dir,
         epochs=1,
         batch_size=128,
         checkpoint_dir="checkpoints",
         checkpoint_every=200,
         empty_cache_every=50,
         num_workers=1,
         resume=True):

    # Optional: when debugging long CUDA issues, set this env var in shell:
    # CUDA_LAUNCH_BLOCKING=1 python train_cvae.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = LazySpeakerFeatureDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = CVAE(num_speakers=len(dataset.speaker_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    start_batch = 0

    # Resume from latest checkpoint if available
    if resume:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest:
            print("Resuming from checkpoint:", latest)
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0)
            start_batch = ckpt.get("batch", 0)
            print(f"Resumed at epoch {start_epoch}, batch {start_batch}")

    # Training loop
    try:
        for epoch in range(start_epoch, epochs):
            train_epoch.current_epoch = epoch + 1
            print(f"Epoch {epoch+1}/{epochs}")
            avg_loss = train_epoch(model, dataloader, optimizer, device,
                                   start_batch=start_batch,
                                   checkpoint_dir=checkpoint_dir,
                                   checkpoint_every=checkpoint_every,
                                   empty_cache_every=empty_cache_every)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

            # Save epoch checkpoint
            ckpt_name = f"cvae_checkpoint_epoch_{epoch+1}_batch_end.pt"
            state = {
                "epoch": epoch + 1,
                "batch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir, ckpt_name)

            # reset start_batch after first resumed epoch
            start_batch = 0

    except Exception as e:
        # Save a safe checkpoint on exception
        print("Exception during training:", repr(e))
        safe_name = f"cvae_safecrash_epoch_{train_epoch.current_epoch}_batch_{start_batch}.pt"
        try:
            state = {
                "epoch": train_epoch.current_epoch,
                "batch": start_batch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir, safe_name)
            print("Saved safe checkpoint before exiting.")
        except Exception as ex2:
            print("Failed to save checkpoint on crash:", repr(ex2))
        raise  # re-raise so you see the original error

if __name__ == "__main__":
    # Tune these params as needed
    main(
        data_dir="/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/train_mini",
        epochs=1,
        batch_size=128,
        checkpoint_dir="checkpoints",
        checkpoint_every=200,     # save every 2500 batches
        empty_cache_every=50,     # call empty_cache every 500 batches
        num_workers=1,
        resume=True
    )
