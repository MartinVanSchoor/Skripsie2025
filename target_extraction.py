import torch
import torchaudio
import torchaudio.functional as F
import time
import argparse
from pathlib import Path
from tqdm import tqdm

def largest_divisor_in_range(n, low=1, high=1_000_000):
    for d in range(high, low - 1, -1):
        if n % d == 0:
            return d

def main(target_length, subpath):
### Laptop
    # dev = "cpu"
    # target_dir_og = Path("/home/martinvs/librispeech_data/LibriSpeech/train-clean-100")
    # target_dir_new = Path(f"/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/train_100")
### Desktop
    dev = "cuda"
    target_dir_og = Path("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/librispeech/Librispeech/test-clean")
    target_dir_new = Path("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/librispeech_target_feats/test/180")
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=dev)


    for speaker_dir in tqdm(sorted(target_dir_og.iterdir()), desc="Processing speakers"):
        if not speaker_dir.is_dir():
            continue

        speaker_name = speaker_dir.name
        speaker_out_dir = target_dir_new / speaker_name
        out_path = speaker_out_dir / f"{speaker_name}.pt"

        if out_path.exists():
            print(f"Skipping {speaker_name}, already processed.")
            continue

        audio = torch.empty(1, 0)
        
        flac_files = sorted(speaker_dir.rglob("*.flac"))
        
        for flac_path in flac_files:
            waveform, sr = torchaudio.load(flac_path)
            audio = torch.cat([audio, waveform], dim=1)
            # For specific amount of audio
            if audio.shape[1] >= target_length:
                audio = audio[:, :target_length]
                break
        length = audio.shape[1] / 960000
        print(f"Accumulated {length} mins of audio") 

        start = time.time()
        audio = audio.to(dev)
        chunk_length = 1440000
        chunk_list = []
        for i in range(0, audio.shape[1], chunk_length):
            chunk = audio[:, i : i + chunk_length]

            # Check if chunk is too short
            if chunk.shape[1] < 10:
                print(f"Skipping chunk of size {chunk.shape[1]} from speaker {speaker_name}")
                continue

            try:
                with torch.inference_mode():
                    chunk_features, _ = wavlm.extract_features(chunk, output_layer=6)
                if chunk_features.dim() == 3:
                    chunk_features = chunk_features.squeeze(0)  # remove batch dimension only
                elif chunk_features.dim() == 2:
                    pass  # already fine
                else:
                    print(f"Unexpected shape: {chunk_features.shape}")
                    continue
                chunk_list.append(chunk_features)
            except Exception as e:
                print(f"Failed to process chunk from speaker {speaker_name}, size={chunk.shape[1]}: {e}")
                continue
        target_features = torch.cat(chunk_list, dim=0)
        print(target_features.shape)
        print(f"Extracted {target_features.shape[0]} features from speaker {speaker_name}")
        print(f"Extraction took {time.time() - start:.4f} seconds")
        
        speaker_out_dir = target_dir_new / speaker_name
        speaker_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_out_dir / f"{speaker_name}.pt"
        torch.save(target_features, out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM features for speakers.")
    parser.add_argument(
        "--target_length",
        type=int,
        default=2880000,
        help="Target length (in samples) of audio to extract features from (default: 2800000)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="180",
        help="Subfolder name under 'data/similarity/' to save extracted features (default: '180')"
    )
    args = parser.parse_args()
    main(args.target_length, args.path)
