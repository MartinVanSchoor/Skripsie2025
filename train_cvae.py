"""
train_cvae.py

Training script for the CVAE model implemented in cvae_model.py.

Top-level global variables (edit these to configure training):
    DATA_DIR         - path to train_dir (contains per-speaker subfolders)
    CHECKPOINT_DIR   - where to save epoch checkpoints
    STATS_PATH       - where to save global normalization stats (mean/std)
    NUM_EPOCHS       - number of training epochs
    BATCH_SIZE       - batch size
    LR               - learning rate
    Z_DIM            - latent dimension
    DEVICE           - 'cuda' or 'cpu' (auto-detected)
    NUM_WORKERS      - DataLoader workers (you asked for 2)
    SAVE_EVERY_EPOCH - whether to save every epoch (True per your request)
    KL_WEIGHT        - beta weight on KL
    ANNEAL_KL        - whether to linearly anneal KL from 0->KL_WEIGHT over epochs
"""

import os
import math
import glob
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

from cvae_model import CVAE

# -------------------------
# Global config (edit here)
# -------------------------
DATA_DIR = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train_mini"           # <-- change this to your data folder
CHECKPOINT_DIR = "checkpoints"
STATS_PATH = os.path.join(CHECKPOINT_DIR, "train_stats.npz")

NUM_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-4
Z_DIM = 128
COND_PROJ_DIM = 256
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY_EPOCH = True
KL_WEIGHT = 1.0
ANNEAL_KL = True
ANNEAL_END_EPOCH = 50  # linearly increase KL weight until this epoch
PRINT_EVERY = 50
SEED = 42
NUM_SAMPLES_TO_LOG = 8  # number of reconstructions to print/save each epoch (if you want)
# -------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class SpeakerFrameDataset(Dataset):
    """
    Dataset that loads all speaker frames into memory and returns individual frames
    with their associated speaker conditioning vector.

    Expects directory structure:
        DATA_DIR/
            speakerA/
                speakerA.npy            # shape (N_frames, x_dim)
                speakerA_id.npy         # shape (c_dim,)
            speakerB/
                ...
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.speakers = sorted([d for d in os.listdir(root_dir)
                                if os.path.isdir(os.path.join(root_dir, d))])
        if not self.speakers:
            raise RuntimeError(f"No speaker directories found in {root_dir}")
        self._load_all()

    def _load_all(self):
        self.frames = []  # list of ndarray (N_i, x_dim)
        self.c_vecs = []  # list of ndarray (c_dim,)
        self.idx_map = []  # list of (speaker_idx, frame_idx)
        self.x_dim = None
        self.c_dim = None

        for spk_idx, spk in enumerate(self.speakers):
            folder = os.path.join(self.root_dir, spk)
            x_paths = glob.glob(os.path.join(folder, "*.npy"))
            # Identify the main feature and id files
            feat_path = None
            id_path = None
            for p in x_paths:
                name = os.path.basename(p)
                if name.endswith("_id.npy") or name.endswith("id.npy"):
                    id_path = p
                else:
                    # assume the other .npy is features
                    if feat_path is None:
                        feat_path = p
            if feat_path is None or id_path is None:
                # try using naming speakername.npy and speakername_id.npy explicitly
                cand_feat = os.path.join(folder, f"{spk}.npy")
                cand_id = os.path.join(folder, f"{spk}_id.npy")
                if os.path.exists(cand_feat) and os.path.exists(cand_id):
                    feat_path = cand_feat
                    id_path = cand_id
                else:
                    raise RuntimeError(f"Could not find feature/id files in {folder}")

            x = np.load(feat_path)  # shape (N_frames, x_dim)
            c = np.load(id_path)    # shape (c_dim,)
            assert x.dtype == np.float32, f"Expect float32 features but got {x.dtype}"
            assert c.dtype == np.float32, f"Expect float32 speaker ids but got {c.dtype}"

            if self.x_dim is None:
                self.x_dim = x.shape[1]
            else:
                if x.shape[1] != self.x_dim:
                    raise RuntimeError("Inconsistent x_dim across speakers")

            if self.c_dim is None:
                self.c_dim = c.shape[0]
            else:
                if c.shape[0] != self.c_dim:
                    raise RuntimeError("Inconsistent c_dim across speakers")

            n_frames = x.shape[0]
            # store
            self.frames.append(x)
            self.c_vecs.append(c)
            # extend idx_map
            for i in range(n_frames):
                self.idx_map.append((spk_idx, i))

        self.total_frames = len(self.idx_map)
        # Optionally stack frames if memory allows; keep per-speaker arrays for indexing.
        print(f"Loaded {len(self.speakers)} speakers, total frames: {self.total_frames}, "
              f"x_dim={self.x_dim}, c_dim={self.c_dim}")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        spk_idx, frame_idx = self.idx_map[idx]
        x = self.frames[spk_idx][frame_idx]
        c = self.c_vecs[spk_idx]
        # return as float32 numpy arrays (DataLoader will convert to tensors)
        return x, c


def compute_dataset_stats(dataset: SpeakerFrameDataset, stats_path: str):
    """
    Compute dataset mean and std over all frames (per-feature).
    Saves to stats_path as numpy .npz with 'mean' and 'std'.
    """
    print("Computing dataset mean/std...")
    # streaming compute mean/std to avoid huge memory copy
    n = 0
    mean = np.zeros(dataset.x_dim, dtype=np.float64)
    m2 = np.zeros(dataset.x_dim, dtype=np.float64)  # for variance (Welford)
    for spk_frames in dataset.frames:
        # spk_frames: (N_i, x_dim)
        for row in spk_frames:
            n += 1
            delta = row - mean
            mean += delta / n
            delta2 = row - mean
            m2 += delta * delta2
    var = m2 / max(1, n-1)
    std = np.sqrt(var).astype(np.float32)
    mean = mean.astype(np.float32)
    # protect small std
    std[std < 1e-6] = 1.0
    np.savez(stats_path, mean=mean, std=std)
    print(f"Saved stats to {stats_path}")
    return mean, std


def load_stats_or_compute(dataset: SpeakerFrameDataset, stats_path: str):
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        mean = data["mean"]
        std = data["std"]
        print(f"Loaded stats from {stats_path}")
    else:
        mean, std = compute_dataset_stats(dataset, stats_path)
    return mean, std


def collate_fn(batch):
    xs = [torch.from_numpy(item[0]) for item in batch]  # (x_dim,)
    cs = [torch.from_numpy(item[1]) for item in batch]
    x = torch.stack(xs, dim=0)
    c = torch.stack(cs, dim=0)
    return x, c


def kl_divergence(mu, logvar):
    # KL between N(mu, sigma^2) and N(0,1): 0.5 * sum( mu^2 + sigma^2 - 1 - log(sigma^2) )
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (batch,)


def train():
    print(f"Using device: {DEVICE}")
    dataset = SpeakerFrameDataset(DATA_DIR)
    mean, std = load_stats_or_compute(dataset, STATS_PATH)

    # normalization tensors (will be applied to x)
    mean_t = torch.from_numpy(mean).float().to(DEVICE)
    std_t = torch.from_numpy(std).float().to(DEVICE)

    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            collate_fn=collate_fn)

    model = CVAE(x_dim=dataset.x_dim,
                 c_dim=dataset.c_dim,
                 cond_proj_dim=COND_PROJ_DIM,
                 z_dim=Z_DIM,
                 enc_hidden=[1024, 512],
                 dec_hidden=[512, 1024],
                 dropout=0.0)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    global_step = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        it = 0

        # compute current KL weight (annealing)
        if ANNEAL_KL:
            kl_w = KL_WEIGHT * min(1.0, epoch / max(1, ANNEAL_END_EPOCH))
        else:
            kl_w = KL_WEIGHT

        for batch_idx, (x_np, c_np) in enumerate(dataloader):
            it += 1
            # to device
            x = x_np.to(DEVICE, non_blocking=True).float()
            c = c_np.to(DEVICE, non_blocking=True).float()

            # normalize
            x = (x - mean_t) / std_t

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            # reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon_x, x, reduction='none')
            recon_loss = recon_loss.mean(dim=1)  # per-sample mse
            recon_loss = recon_loss.mean()       # scalar

            kl = kl_divergence(mu, logvar).mean()  # scalar

            loss = recon_loss + kl_w * kl

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl.item()
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                print(f"Epoch {epoch} Step {global_step} Loss {loss.item():.6f} (recon {recon_loss.item():.6f}, kl {kl.item():.6f}, kl_w {kl_w:.4f})")

        avg_loss = epoch_loss / max(1, it)
        avg_recon = epoch_recon / max(1, it)
        avg_kl = epoch_kl / max(1, it)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{NUM_EPOCHS}  avg_loss={avg_loss:.6f}  recon={avg_recon:.6f}  kl={avg_kl:.6f}  kl_w={kl_w:.4f}  elapsed={elapsed/60:.2f}m")

        # Save checkpoint every epoch (as requested)
        if SAVE_EVERY_EPOCH:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "mean": mean,
                "std": std,
                "config": {
                    "x_dim": dataset.x_dim,
                    "c_dim": dataset.c_dim,
                    "z_dim": Z_DIM,
                    "cond_proj_dim": COND_PROJ_DIM
                }
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"cvae_epoch{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    train()
