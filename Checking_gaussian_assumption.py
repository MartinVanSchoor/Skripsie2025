import os, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import wasserstein_distance

def wasserstein_distance_normality_check(X):
        X = np.asarray(X)
        n, d = X.shape
        # compare each column to fitted Normal N(mu, sigma^2); average 1D WDs
        vals = []
        for j in range(d):
            col = X[:, j]
            mu, sig = col.mean(), col.std() + 1e-8
            ref = np.random.normal(mu, sig, size=n)
            vals.append(wasserstein_distance(col, ref))
        return float(np.mean(vals))

# Load train-clean-100 speaker features dictionary: {"id": str, "feats": torch.Tensor(T, 1024)}
FEAT_DIR = "C:/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train-100-feats"   # change if you saved elsewhere
assert os.path.isdir(FEAT_DIR), f"Feature dir not found: {FEAT_DIR}"

feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.pt")))
assert feat_files, f"No .pt features found in {FEAT_DIR}. Run the extraction step first."

# ---- build query_seq from precomputed features (no wavs, no model calls) ----
N_SAMPLES = min(10, len(feat_files))   # match your loop count
rng = np.random.default_rng(42)
pick_idx = rng.choice(len(feat_files), size=N_SAMPLES, replace=False)
pick_files = [feat_files[i] for i in pick_idx]

query_seq_list = []
for path in tqdm(pick_files):
    pack = torch.load(path, map_location="cpu")
    feats = pack["feats"]  # shape (T, D)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    query_seq_list.append(feats)

query_seq = torch.cat(query_seq_list, dim=0)  # (sum_T, D)
embed_len = query_seq.shape[1]
print("query_seq shape:", tuple(query_seq.shape))

# ---- your ranking + array for tests ----
idxes = torch.argsort(query_seq.std(dim=0), descending=True)
data_for_test = query_seq[:, idxes].cpu().numpy()
print("data_for_test shape:", data_for_test.shape)

# ---- compute W distances exactly like your loop (over windows of dims) ----
dimensions = [2**i for i in [2, 4, 6, 8]]  # [4, 16, 64, 256]
step = 8
x_axis_indices = list(range(0, embed_len, step))

wass_data = []
for D in tqdm(dimensions, desc="Processing dimensions"):
    wasserstein_distance_list = []
    for idx in tqdm(x_axis_indices, desc=f"Dimension {D}", leave=False):
        # allow the last window to be shorter than D (keeps behavior similar to your original)
        wasserstein_distance_list.append(
            wasserstein_distance_normality_check(data_for_test[:, idx:idx + D])
        )
    wass_data.append(wasserstein_distance_list)

# ---- plot ----
plt.figure(figsize=(8, 4.75))
colors = ['r', 'g', 'b', 'orange']
for i, D in enumerate(dimensions):
    y = np.array(wass_data[i])
    x = x_axis_indices[:len(y)]
    plt.plot(x, y, label=f"MKL dim $K$={D}", color=colors[i], linewidth=2, marker='o', markersize=4)

plt.title('Proof of the Gaussian assumption for WavLM features', fontsize=15, fontweight="bold")
plt.xlabel('Starting Index of WavLM Embedding Dim', fontsize=14)
plt.ylabel('Wasserstein Distance / (MKL dim)', fontsize=14)
plt.xticks(fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7, which='major')
plt.grid(True, linestyle=':', alpha=0.4, which='minor')
plt.legend(loc='upper right', fontsize=16)
plt.tight_layout()
plt.savefig('wasserstein_distance_plot.pdf')
plt.show()