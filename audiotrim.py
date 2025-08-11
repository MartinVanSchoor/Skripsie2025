from pydub import AudioSegment
import torch
from pathlib import Path
from tqdm import tqdm

dir_old = Path("C:/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/train_reduced")
dir_new = Path("C:/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/train_mini")
count = 0

for speaker_dir in tqdm(sorted(dir_old.iterdir()), desc="Processing speakers"):
    if not speaker_dir.is_dir():
            continue

    filename = f"{speaker_dir.name}.pt"
    path_old = speaker_dir / filename
    speaker_path_new = dir_new / speaker_dir.name
    path_new = speaker_path_new / f"{speaker_dir.name}.pt"
    if (count % 5 == 0):
        speaker_path_new.mkdir(parents=True, exist_ok=True)
        tensor = torch.load(path_old, map_location=torch.device('cpu'))
        torch.save(tensor, path_new)
    count = count + 1
