# train_cvae.py (fault-tolerant, resumable, robust checkpointing)
import os
import glob
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

        # Lazy load
        features = np.load(feature_path, mmap_mode=None)
        feature_vec = torch.from_numpy(features[feature_idx]).float()
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
# Training loop
# ---------------------------
def train_epoch(model, dataloader, optimizer, device, current_epoch_idx, total_epochs,
                start_batch=0, checkpoint_dir="checkpoints",
                checkpoint_every=100, empty_cache_every=50):
    """
    current_epoch_idx: zero-based epoch index
    start_batch: zero-based index of batch to start from (0 means start of epoch)
    """
    model.train()
    total_loss = 0.0
    batch_idx = -1

    human_epoch = current_epoch_idx + 1
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {human_epoch}/{total_epochs} - Training",
                initial=start_batch)

    for i, (x, speaker_ids) in pbar:
        if i < start_batch:
            continue

        x = x.to(device, non_blocking=True).float()
        speaker_ids = speaker_ids.to(device, non_blocking=True)

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
        pbar.set_postfix(loss=loss.item())

        if (i + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

        # periodic checkpoint: save resume_epoch_idx = current_epoch_idx, resume_batch = i+1
        if (i + 1) % checkpoint_every == 0:
            ckpt_name = f"cvae_checkpoint_epoch_{human_epoch}_batch_{i+1}.pt"
            state = {
                "epoch": human_epoch,               # human readable (1-based)
                "epoch_idx": current_epoch_idx,    # zero-based epoch index for this epoch
                "batch": i + 1,                    # next batch index to resume at
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir, ckpt_name)

    # if no batches processed (batch_idx == -1) we return 0
    if batch_idx < 0:
        return 0.0
    return total_loss / max(1, (batch_idx + 1))

# ---------------------------
# Main
# ---------------------------
def main(data_dir,
         epochs=5,
         batch_size=128,
         checkpoint_dir="checkpoints",
         checkpoint_every=100,
         empty_cache_every=50,
         num_workers=1,
         resume=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = LazySpeakerFeatureDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)

    model = CVAE(num_speakers=len(dataset.speaker_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # start_epoch is zero-based index
    start_epoch = 0
    start_batch = 0

    # Resume from latest checkpoint if available
    if resume:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest:
            print("Resuming from checkpoint:", latest)
            ckpt = torch.load(latest, map_location=device)
            # load weights/state
            model.load_state_dict(ckpt.get("model_state", model.state_dict()))
            optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))

            # Backwards-compatible resume parsing:
            # Prefer explicit zero-based 'epoch_idx' if present. Otherwise use heuristic from ('epoch','batch').
            if "epoch_idx" in ckpt:
                start_epoch = int(ckpt.get("epoch_idx", 0))
                start_batch = int(ckpt.get("batch", 0)) if ckpt.get("batch", None) is not None else 0
            else:
                # legacy: ckpt['epoch'] is human-readable (1-based), ckpt['batch'] is next batch
                ckpt_epoch = int(ckpt.get("epoch", 0))
                ckpt_batch = int(ckpt.get("batch", 0)) if ckpt.get("batch", None) is not None else 0
                if ckpt_batch == 0:
                    # completed ckpt_epoch, start at next epoch index (zero-based)
                    start_epoch = ckpt_epoch
                    start_batch = 0
                else:
                    # mid-epoch checkpoint where ckpt_epoch is human (1-based)
                    start_epoch = max(ckpt_epoch - 1, 0)
                    start_batch = ckpt_batch

            # safety clamp
            if start_epoch >= epochs:
                print(f"Checkpoint indicates start_epoch {start_epoch} >= configured epochs {epochs}. Setting start_epoch={epochs-1}, start_batch=0")
                start_epoch = max(0, epochs - 1)
                start_batch = 0

            print(f"Resuming at (human) epoch {start_epoch+1}, batch {start_batch}")

    # Training loop
    try:
        for epoch_idx in range(start_epoch, epochs):
            human_epoch = epoch_idx + 1
            print(f"Epoch {human_epoch}/{epochs}")
            avg_loss = train_epoch(model, dataloader, optimizer, device,
                                   current_epoch_idx=epoch_idx, total_epochs=epochs,
                                   start_batch=start_batch if epoch_idx == start_epoch else 0,
                                   checkpoint_dir=checkpoint_dir,
                                   checkpoint_every=checkpoint_every,
                                   empty_cache_every=empty_cache_every)
            print(f"Epoch {human_epoch} average loss: {avg_loss:.6f}")

            # Save epoch-complete checkpoint: indicate next epoch index = epoch_idx + 1 and batch = 0
            ckpt_name = f"cvae_checkpoint_epoch_{human_epoch}_batch_end.pt"
            state = {
                "epoch": human_epoch,
                "epoch_idx": epoch_idx + 1,  # next epoch index (zero-based)
                "batch": 0,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir, ckpt_name)

            # reset start_batch after we've resumed the first epoch
            start_batch = 0

    except Exception as e:
        print("Exception during training:", repr(e))
        # Try to save a safe checkpoint using the best-known indices
        try:
            safe_name = f"cvae_safecrash_epoch_{start_epoch}_batch_{start_batch}.pt"
            state = {
                "epoch": start_epoch + 1,
                "epoch_idx": start_epoch,
                "batch": start_batch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir, safe_name)
            print("Saved safe checkpoint before exiting.")
        except Exception as ex2:
            print("Failed to save checkpoint on crash:", repr(ex2))
        raise

if __name__ == "__main__":
    main(
        data_dir="/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/train_mini",
        epochs=5,
        batch_size=128,
        checkpoint_dir="checkpoints",
        checkpoint_every=100,
        empty_cache_every=50,
        num_workers=1,
        resume=True
    )
