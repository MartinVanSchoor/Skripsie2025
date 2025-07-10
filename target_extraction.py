import torch
import torchaudio
import torchaudio.functional as F

target_fn = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target1_trump.wav"

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

# Test for one chunk


# Divide the target audio into chunks, extract the features, append and store in a .npy file
chunk_length = target_audio.shape[1] // 36
for i in range(36):
    chunk = target_audio[:,(i*chunk_length):((i+1)*chunk_length)]
    print(chunk.shape)


