import numpy as np
import scipy
import torch
import torchaudio
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Load in the neccessary models (SSL and Vocoder)
device = "cpu"
wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
print("Great success")