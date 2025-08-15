"""
sample_from_cvae.py

Utilities to generate (sample) new feature frames from a trained CVAE checkpoint.

Two modes:
1) Prior sampling:
    - Provide a speaker_id.npy file (shape (512,)), and sample z ~ N(0, I), decode many times.
2) Posterior (seed) sampling:
    - Provide one or more seed frames (seed.npy of shape (N_seed, x_dim)) and a speaker_id
      The script encodes the seed frames (with the provided speaker id),
      averages the posterior mu/logvar across the seeds to obtain an estimated posterior,
      then samples z from that posterior and decodes many times to create more frames.

Edit the global fields below to set file paths and generation parameters, or use
these functions programmatically.
"""

import os
import numpy as np
import torch

from cvae_model import CVAE


# ---------------------------
# Edit these when running CLI
# ---------------------------
CHECKPOINT_PATH = "checkpoints/cvae_epoch200.pt"  # checkpoint to load
SPEAKER_ID_PATH = "some_speaker/some_speaker_id.npy"  # required
SEED_FRAMES_PATH = None  # optional: path to .npy of shape (N_seed, x_dim). If None -> prior sampling
N_SAMPLES = 8847
OUTPUT_PATH = "generated_samples.npy"  # saves (N_SAMPLES, x_dim)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------


def load_checkpoint(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    x_dim = cfg.get("x_dim", 1024)
    c_dim = cfg.get("c_dim", 512)
    z_dim = cfg.get("z_dim", 128)
    cond_proj_dim = cfg.get("cond_proj_dim", 256)
    model = CVAE(x_dim=x_dim, c_dim=c_dim, cond_proj_dim=cond_proj_dim, z_dim=z_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    return model, mean, std


def sample_prior(model, c_vec, n_samples=100, mean=None, std=None, device="cpu"):
    """
    Sample z ~ N(0,I) and decode conditioned on repeated c_vec.
    c_vec: np.ndarray shape (c_dim,)
    """
    c = torch.from_numpy(c_vec).float().unsqueeze(0).to(device)  # (1, c_dim)
    # repeat for n_samples
    c_rep = c.repeat(n_samples, 1)
    with torch.no_grad():
        x_norm = model.sample_prior(n_samples, c=c_rep, device=device)  # normalized space
        x_norm = x_norm.cpu().numpy()
    if mean is not None and std is not None:
        # de-normalize
        x = x_norm * std[np.newaxis, :] + mean[np.newaxis, :]
    else:
        x = x_norm
    return x


def sample_from_seed_posterior(model, seed_frames, c_vec, n_samples=100, mean=None, std=None, device="cpu"):
    """
    Encode seed_frames with c_vec to get posterior parameters.
    seed_frames: np.ndarray shape (N_seed, x_dim)
    Steps:
        - normalize seed frames with mean/std (if provided)
        - compute mu, logvar for each seed frame
        - average mu and logvar across seeds
        - sample n_samples z from averaged posterior
        - decode each z with the provided c_vec (repeated)
    """
    x = torch.from_numpy(seed_frames).float().to(device)
    c = torch.from_numpy(c_vec).float().unsqueeze(0).to(device)  # (1, c_dim)
    # repeat c for each seed frame to encode
    c_rep = c.repeat(x.shape[0], 1)
    if mean is not None and std is not None:
        mean_t = torch.from_numpy(mean).float().to(device)
        std_t = torch.from_numpy(std).float().to(device)
        x = (x - mean_t) / std_t
    with torch.no_grad():
        mu, logvar = model.encode(x, c_rep)  # (N_seed, z_dim)
        # average
        mu_avg = mu.mean(dim=0, keepdim=True)       # (1, z_dim)
        logvar_avg = logvar.mean(dim=0, keepdim=True)
        std_avg = torch.exp(0.5 * logvar_avg)
        # sample n_samples z
        eps = torch.randn(n_samples, mu_avg.shape[1], device=device)
        z = mu_avg.repeat(n_samples, 1) + eps * std_avg.repeat(n_samples, 1)
        c_rep_samples = c.repeat(n_samples, 1)
        x_norm = model.decode(z, c_rep_samples)  # (n_samples, x_dim)
        x_norm = x_norm.cpu().numpy()
    if mean is not None and std is not None:
        x = x_norm * std[np.newaxis, :] + mean[np.newaxis, :]
    else:
        x = x_norm
    return x


def main():
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    assert os.path.exists(SPEAKER_ID_PATH), f"Speaker id file not found: {SPEAKER_ID_PATH}"

    model, mean, std = load_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    c_vec = np.load(SPEAKER_ID_PATH).astype(np.float32)

    if SEED_FRAMES_PATH is None:
        print(f"Sampling {N_SAMPLES} frames from prior conditioned on speaker id")
        samples = sample_prior(model, c_vec, n_samples=N_SAMPLES, mean=mean, std=std, device=DEVICE)
    else:
        assert os.path.exists(SEED_FRAMES_PATH), f"Seed frames file not found: {SEED_FRAMES_PATH}"
        seed_frames = np.load(SEED_FRAMES_PATH).astype(np.float32)
        print(f"Using {seed_frames.shape[0]} seed frames to estimate posterior and sampling {N_SAMPLES} frames.")
        samples = sample_from_seed_posterior(model, seed_frames, c_vec, n_samples=N_SAMPLES, mean=mean, std=std, device=DEVICE)

    # Save generated features
    np.save(OUTPUT_PATH, samples)
    print(f"Saved generated samples to {OUTPUT_PATH} (shape: {samples.shape})")


if __name__ == "__main__":
    main()
