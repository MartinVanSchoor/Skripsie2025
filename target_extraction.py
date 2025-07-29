import torch
import torchaudio
import torchaudio.functional as F
from transformers import WavLMModel, WavLMPr
import time
from pathlib import Path
from tqdm import tqdm

def largest_divisor_in_range(n, low=1, high=800_000):
    for d in range(high, low - 1, -1):
        if n % d == 0:
            return d

# Define variables and import WavLM-large
dev = "cuda"
target_dir_og = Path("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/librispeech/LibriSpeech/test-clean")
target_dir_new = Path("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/Skripsie2025_desktop/data/targets_librispeech/180")
wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=dev)

# Loop through each speaker
for speaker_dir in tqdm(sorted(target_dir_og.iterdir()), desc="Processing speakers"):
    if not speaker_dir.is_dir():
        continue

    speaker_name = speaker_dir.name
    audio = torch.empty(1, 0)  # shape: [1, 0]
    print(speaker_name)

# # Retrieve audio
# target_audio, sr = torchaudio.load("/mnt/c/Users/Martin/Documents/Werk/Universiteit/Skripsie_desktop/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac")
# print(target_audio.shape)
# # convert to appropriate device
# target_audio = target_audio.to(dev)

# start = time.time()
# # Divide the target audio into chunks, extract the features, append and store in a .npy file
# chunk_length = largest_divisor_in_range(target_audio.shape[1])
# print(f"Chunk length = {chunk_length}")
# chunk_list = []
# for i in range(target_audio.shape[1]//chunk_length):
#     chunk = target_audio[:,(i*chunk_length):((i+1)*chunk_length)]
#     with torch.inference_mode():
#         chunk_features, _ = wavlm.extract_features(chunk, output_layer=6)
#     chunk_features = chunk_features.squeeze()
#     chunk_list.append(chunk_features)
# target_features = torch.cat(chunk_list, dim=0)
# print(target_features.shape)
# end = time.time()
# print(f"Extraction took {end - start:.4f} seconds")
    

