from pathlib import Path
from tqdm import tqdm
import numpy as np

dir = Path("/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train_mini")

for speaker_dir in tqdm(sorted(dir.iterdir()), desc="Processing speakers"):
    if not speaker_dir.is_dir():
        continue

    feat_fn = f"{speaker_dir.name}.npy"
    feat_path_old = speaker_dir / feat_fn
    id_fn = f"{speaker_dir.name}_id.npy"
    id_path_old = speaker_dir / id_fn

    features = np.load(feat_path_old)
    ids = np.load(id_path_old)

    print(features.dtype)
    print(ids.dtype)