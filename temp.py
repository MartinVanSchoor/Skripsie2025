from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch 

dir = Path("/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train_100")
dir_new = Path("/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train-100-feats")

for speaker_dir in tqdm(sorted(dir.iterdir()), desc="Processing speakers"):
    if not speaker_dir.is_dir():
        continue

    feat_path_old = speaker_dir / f"{speaker_dir.name}.npy"
    feat_path_new = dir_new / f"{speaker_dir.name}.pt"
    print(feat_path_new)
    # id_fn = f"{speaker_dir.name}_id.pt"
    # id_path_old = speaker_dir / id_fn

    features = np.load(feat_path_old)
    features = torch.from_numpy(features)
    print(features.shape)
    # ids = torch.load(id_path_old)

    torch.save({"id": speaker_dir.name, "feats": features.cpu()}, feat_path_new)