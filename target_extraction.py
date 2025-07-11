import torch
import torchaudio
import torchaudio.functional as F
import time

target_fn = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target1_trump.wav"
wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device="cpu")

# Retrieve audio
target_audio, sr = torchaudio.load(target_fn)
print(target_audio.shape)
# Convert to mono if stereo
if target_audio.shape[0] > 1:
    target_audio = target_audio.mean(dim=0, keepdim=True)
print(target_audio.shape)
# Resample to 16kHz if needed
if not sr == 16000:
    target_audio = F.resample(
        target_audio,
        orig_freq=sr,
        new_freq=16000,
    )
# Convert to appropriate device
target_audio = target_audio.to("cpu")
print(target_audio.shape)

start = time.time()
# Divide the target audio into chunks, extract the features, append and store in a .npy file
div = 32
chunk_length = target_audio.shape[1] // div
chunk_list = []
for i in range(div):
    chunk = target_audio[:,(i*chunk_length):((i+1)*chunk_length)]
    with torch.inference_mode():
        chunk_features, _ = wavlm.extract_features(chunk, output_layer=6)
    chunk_features = chunk_features.squeeze()
    chunk_list.append(chunk_features)
target_features = torch.cat(chunk_list, dim=0)
print(target_features.shape)
end = time.time()
print(f"Extraction took {end - start:.4f} seconds")
    

