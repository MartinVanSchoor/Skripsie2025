import numpy as np
import scipy
import torch
import torchaudio
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import torchaudio.functional as F

n_frames = None
k_top = 4

class kNN_VC(torch.nn.Module):
    def __init__(self, wavlm, hifigan, device="cpu"):
        super().__init__()
        self.wavlm = wavlm.eval()
        self.hifigan = hifigan.eval()
        self.device = device
        
    @torch.inference_mode()
    def get_features(self, audio_fn):
        """
        Returns features from file specified by audio_fn as a tensor with 1024 dimensions
        VAD can be applie if vad=True
        """
        # Retrieve audio and convert to correct device
        source_audio, _ = torchaudio.load(audio_fn)
        source_audio = source_audio.to(self.device)
        
        # Use the WavLM SSL to extract features
        features, _ = self.wavlm.extract_features(source_audio, output_layer=6)
        #features = features.squeeze()
        return features
    
    @torch.inference_mode()
    def vocode(self, input_features):
        """ 
        Returns the waveform samples
        """
        wav_hat = self.hifigan(input_features).squeeze(0)
        return wav_hat
        
        
def main():
    device = "cpu"
    # Load in the neccessary models (SSL feature extractor and Vocoder)
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    print("Great success")

if __name__ == "__main__":
    main()