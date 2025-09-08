from pathlib import Path
from tqdm import tqdm
import numpy as np
import torchaudio
import torch
import torchaudio.functional as F
from knn_vc import kNN_VC
from sklearn.neighbors import NearestNeighbors
from mkl import apply_mkl_batched
        
device = "cuda"
batch_size = 256
dir1 = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/target/louw1.wav"
dir2 = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/target/louw2.wav"
dir3 = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/target/louw3.wav"
source_dir = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/source/source_to_louw.wav"
dir_out = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/output/louw.wav"

wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

clip1, _ = torchaudio.load(dir1)
clip2, _ = torchaudio.load(dir2)
clip3, _ = torchaudio.load(dir3)
clip, sr = torchaudio.load(source_dir)
audio = torch.cat([clip1, clip2, clip3], dim=1)
target = F.resample(audio, orig_freq=sr, new_freq=16000)
source = F.resample(clip, orig_freq=sr, new_freq=16000)
target = target.to(device)
source = source.to(device)

# extract source features
with torch.no_grad():
    source_features, _ = wavlm.extract_features(source, output_layer = 6)
source_features = source_features.squeeze()
print(source_features.shape)

chunk_length = 80000
chunk_list = []
for i in range(target.shape[1]//chunk_length):
    chunk = target[:,(i*chunk_length):((i+1)*chunk_length)]
    with torch.no_grad():
        chunk_features, _ = wavlm.extract_features(chunk, output_layer=6)
    chunk_features = chunk_features.squeeze()
    chunk_list.append(chunk_features)
target_features = torch.cat(chunk_list, dim=0)
print(target_features.shape)

# Convert to numpy for sklearn
source_np = source_features.cpu().numpy()
target_np = target_features.cpu().numpy()
# Fit NearestNeighbors using cosine distance
nn = NearestNeighbors(n_neighbors=4, metric="cosine")
nn.fit(target_np)
# Find 4 nearest neighbors for each source row
distances, indices = nn.kneighbors(source_np)  # indices: (N_source, 4)
# Average the 4 neighbors for each source entry
averaged = np.array([
    target_np[neighbor_indices].mean(axis=0)
    for neighbor_indices in indices
])  
# Convert back to torch
output_features = torch.from_numpy(averaged).to(device) 

wav_hat = hifigan(output_features[None].to(device))
wav_hat = wav_hat.squeeze(1)
output_wav = wav_hat.cpu().squeeze()
torchaudio.save(dir_out, output_wav[None], 16000)