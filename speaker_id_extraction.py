import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path
from tqdm import tqdm


def main(target_length, set):
### Laptop
    # dev = "cpu"
    # librispeech_dir = Path("/home/martinvs/librispeech_data/LibriSpeech/train-clean-100")
    # target_dir = Path(f"/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/train_100")
### Desktop
    dev = "cuda"
    librispeech_dir = Path(f"/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/librispeech/LibriSpeech/{set}-clean")
    target_dir = Path(f"/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/librispeech_targets/{set}/{target_length}")
### Speaker identity model
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": dev},
    )
    
### Extract speaker identity vector for each librispeech train-clean speaker
    for speaker_dir in tqdm(sorted(librispeech_dir.iterdir()), desc="Extracting speaker identity"):
        # Ensure speaker exists
        if not speaker_dir.is_dir():
            continue
        
        # Construct the neccessary directories and name variables
        speaker_name = speaker_dir.name
        speaker_out_dir = target_dir / speaker_name
        out_path = speaker_out_dir / f"{speaker_name}_id.pt"
        
        # Ensure the speaker has not already been processed
        if out_path.exists():
            print(f"Skipping speaker {speaker_name}, already processed")
            continue

        # Get all audio files for speaker
        flac_files = sorted(speaker_dir.rglob("*.flac"))
        
        # Load 5 seconds of audio from speaker
        for flac_path in flac_files:
            audio, _ = torchaudio.load(flac_path)
            if (audio.shape[1] < 80000):
                continue
            else:
                audio = audio[:, :80000]
                break  
        
        # Extract speaker identity vector and save to .pt file
        audio = audio.to(dev)  
        x = classifier.encode_batch(audio).squeeze().cpu()
        speaker_out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(x, out_path)
                  
            
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_length', type=int, default=180, help='Target length for features')
    parser.add_argument('--set', type=str, default="dev", help='Librispeech set to use')
    args = parser.parse_args()
    main(args.target_length, args.set)
        
        
        
