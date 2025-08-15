from pathlib import Path
from tqdm import tqdm
import numpy as np

dir_old = Path("/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train_reduced")
dir_new = Path("/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train_mini")
count = 0

for speaker_dir in tqdm(sorted(dir_old.iterdir()), desc="Processing speakers"):
    if not speaker_dir.is_dir():
        continue

    feat_fn = f"{speaker_dir.name}.npy"
    feat_path_old = speaker_dir / feat_fn
    id_fn = f"{speaker_dir.name}_id.npy"
    id_path_old = speaker_dir / id_fn

    # Load the tensor & convert to numpy
    features = np.load(feat_path_old)
    ids = np.load(id_path_old)

    speaker_dir_new = dir_new / speaker_dir.name
    feat_path_new = speaker_dir_new / f"{speaker_dir.name}.npy"
    id_path_new = speaker_dir_new / f"{speaker_dir.name}_id.npy"
    
    count = count + 1
    if (count % 5 == 0):
        speaker_dir_new.mkdir(parents=True, exist_ok=True)
        np.save(feat_path_new, features)
        np.save(id_path_new, ids)
        
