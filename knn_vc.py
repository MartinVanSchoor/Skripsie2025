import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import torchaudio.functional as F

n_frames = None
k_top = 4

def largest_divisor_in_range(n, low=5000, high=800_000):
    for d in range(high, low - 1, -1):
        if n % d == 0:
            return d

class kNN_VC(torch.nn.Module):
    def __init__(self, wavlm, hifigan, k, device="cpu"):
        super().__init__()
        self.wavlm = wavlm.eval()
        self.hifigan = hifigan.eval()
        self.k = k
        self.device = device 
        self.sr_target = 16000
        
    @torch.inference_mode()
    def get_features(self, audio_fn, mode):
        """
        Returns  SSL features from file specified by audio_fn using WavLM-large
        
        mode = 0: Extract and return target features, using chunking
        mode = 1: Extract and return source features
        """
        ## Asserts start
        assert mode in (0, 1), f'"mode" must be 0 or 1, but got {mode}'
        ## Asserts end
        
        # Retrieve audio
        audio, sr = torchaudio.load(audio_fn)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # Resample to 16kHz if needed
        if not sr == self.sr_target:
            audio = F.resample(
                audio,
                orig_freq=sr,
                new_freq=self.sr_target,
            )
        # Convert to appropriate device
        audio = audio.to(self.device)
        
        # Use the WavLM SSL to extract features
        if mode == 0:
            # Divide the target audio into chunks and extract the features
            chunk_length = largest_divisor_in_range(audio.shape[1])
            chunk_list = []
            for i in range(audio.shape[1]//chunk_length):
                chunk = audio[:,(i*chunk_length):((i+1)*chunk_length)]
                chunk_features, _ = self.wavlm.extract_features(chunk, output_layer=6)
                chunk_features = chunk_features.squeeze()
                chunk_list.append(chunk_features)
            target_features = torch.cat(chunk_list, dim=0)
            return target_features
        elif mode == 1:
            # Extract the features from the source audio
            source_features, _ = self.wavlm.extract_features(audio, output_layer=6)
            source_features = source_features.squeeze()
            return source_features
    
    @torch.inference_mode()
    def vocode(self, output_features):
        """ 
        Returns the waveform samples using hifigan vocoder
        """
        wav_hat = self.hifigan(output_features)
        wav_hat = wav_hat.squeeze(1)
        return wav_hat
    
    @torch.inference_mode()
    def knn_matching(self, source_feats, target_feats):
        """ 
        Performs kNN matching and returns the output features
        """
        # Convert to numpy for sklearn
        source_np = source_feats.cpu().numpy()
        target_np = target_feats.cpu().numpy()
        # Fit NearestNeighbors using cosine distance
        nn = NearestNeighbors(n_neighbors=self.k, metric="cosine")
        nn.fit(target_np)
        # Find 4 nearest neighbors for each source row
        distances, indices = nn.kneighbors(source_np)  # indices: (N_source, 4)
        # Average the 4 neighbors for each source entry
        averaged = np.array([
            target_np[neighbor_indices].mean(axis=0)
            for neighbor_indices in indices
        ])  
        # Convert back to torch
        output_features = torch.from_numpy(averaged).to(self.device)   
        return output_features  
        
def main():
    # Specify filenames and other variables
    device = "cpu"
    source_wav_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source1_martin.wav"
    target_wav_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target1_trump.wav"
    output_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/output1.wav"
    # Load in the neccessary models (SSL feature extractor and Vocoder)
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    
    # Extract the target features 
    vc_model = kNN_VC(wavlm, hifigan, k_top, device)
    target_features = vc_model.get_features(target_wav_filename, mode=0)
    print(target_features.shape)
    
    # Extract the source features
    source_features = vc_model.get_features(source_wav_filename, mode=1)
    print(source_features.shape)
    
    # Perform kNN matching to get output features
    output_features = vc_model.knn_matching(source_features, target_features)
    print(output_features.shape)
    
    # Vocode and save the output
    output_wav = vc_model.vocode(output_features[None].to(device)).cpu().squeeze()
    torchaudio.save(output_filename, output_wav[None], vc_model.sr_target)
    
if __name__ == "__main__":
    main()