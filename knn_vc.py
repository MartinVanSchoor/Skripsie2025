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
    def __init__(self, wavlm, hifigan, k, device="cpu"):
        super().__init__()
        self.wavlm = wavlm.eval()
        self.hifigan = hifigan.eval()
        self.k = k
        self.device = device
        
    @torch.inference_mode()
    def get_features(self, audio_fn):
        """
        Returns features from file specified by audio_fn as a tensor with 1024 dimensions
        VAD can be applie if vad=True
        """
        # Retrieve audio and convert to correct device
        source_audio, _ = torchaudio.load(audio_fn)
        print(source_audio.shape)
        #source_audio = source_audio.to(self.device)
        
        # Use the WavLM SSL to extract features
        #features, _ = self.wavlm.extract_features(source_audio, output_layer=6)
        #features = features.squeeze()
        #return features
    
    @torch.inference_mode()
    def vocode(self, input_features):
        """ 
        Returns the waveform samples
        """
        wav_hat = self.hifigan(input_features).squeeze(0)
        return wav_hat.cpu().squeeze().cpu()
    
    @torch.inference_mode()
    def knn_matching(self, source_feats, target_feats):
        """ 
        Performs kNN matching and returns the output features
        """
        nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        nn.fit(target_feats)
        
        
        
def main():
    # Specify filenames and other variables
    device = "cpu"
    source_wav_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source1_martin.wav"
    target_wav_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target1_trump.wav"
    output_path = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/"
    # Load in the neccessary models (SSL feature extractor and Vocoder)
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    print("Models loaded in successfully")

if __name__ == "__main__":
    main()