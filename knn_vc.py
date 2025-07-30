import numpy as np
import torch
import torchaudio
from sklearn.neighbors import NearestNeighbors
import torchaudio.functional as F
import time
import intelligibility
import similarity
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

n_frames = None
k_top = 4

def largest_divisor_in_range(n, low=1, high=800_000):
    for d in range(high, low - 1, -1):
        if n % d == 0:
            return d
        
def evaluate_intelligibility(groundtruth, converted):
    args = SimpleNamespace(
        format="librispeech",
        converted_dir=converted,
        groundtruth_dir=groundtruth,
        whisper="small"
    )
    wer_mean, wer_std, cer_mean, cer_std = intelligibility.main(args)
    return wer_mean, wer_std, cer_mean, cer_std

def evaluate_similarity(groundtruth, converted, eval):
    args = SimpleNamespace(
        format="librispeech",
        eval_csv=eval,
        converted_dir=converted,
        groundtruth_dir=groundtruth,
        zero_positive = 0
    )
    similarity.speaker_similarity(args)

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
        Returns the waveform samples using a pretrained HiFi-GAN vocoder
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
    
### Specify filenames and other variables
 ## For Laptop
    device = "cpu"
    # Target 
    target_wav_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/target/target3_obama.wav"
    target_feat_filename = "data/target/rfk2_5.pt"
    # Source and output, real world
    source_wav_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/source/source3_theo.wav"
    output_filename = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/output/output_TheoToObama.wav"
    # Source and output, intelligibility
    librispeech_dir_I = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/librispeech/Librispeech/test-clean/1089/134686")
    output_dir_I = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/intelligibility/2_5_rfk")
    # Source and output, similarity
    eval_csv = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/eval.csv")
    librispeech_dir_S = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/librispeech/Librispeech/test-clean")
    output_dir_S = Path(f"/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/similarity_libri/180")
    targets_dir = Path(f"/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/librispeech_target_feats/180")
 ## For Desktop
    # device = "cuda"
    
### Load in the neccessary models {SSL feature extractor (WavLM) and Vocoder (HiFi-GAN)}
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    
### Timing start and model initialization
    # start = time.time()
    # vc_model = kNN_VC(wavlm, hifigan, k_top, device)
    
### Target feature extraction/loading
    # Extract the target features from an audio file
    # target_features = vc_model.get_features("data/target/target1_trump2_5.wav", mode=0)
    # torch.save(target_features, "data/target/trump2_5.pt")
    # print(f"Extracted {target_features.shape[0]} features from trump")
    # target_features = vc_model.get_features("data/target/target2_rfk2_5.wav", mode=0)
    # torch.save(target_features, "data/target/rfk2_5.pt")
    # print(f"Extracted {target_features.shape[0]} features from rfk")
    # target_features = vc_model.get_features("data/target/target3_obama2_5.wav", mode=0)
    # torch.save(target_features, "data/target/obama2_5.pt")
    # print(f"Extracted {target_features.shape[0]} features from obama")
    
    # Load the target features from .pt file
    # target_features = torch.load(target_feat_filename)
    # print(f"Loaded {target_features.shape[0]} features from target speaker")
    
### Normal, real-world conversion
    # Extract the source features
    # source_features = vc_model.get_features(source_wav_filename, mode=1)
    # print(f"Extracted {source_features.shape[0]} features from source speaker")
    # print("Features extracted, performing kNN matching...")
    
    # # Perform kNN matching to get output features
    # output_features = vc_model.knn_matching(source_features, target_features)
    # print("Matching complete, vocoding output...")
    
    # # Vocode and save the output
    # output_wav = vc_model.vocode(output_features[None].to(device)).cpu().squeeze()
    # torchaudio.save(output_filename, output_wav[None], vc_model.sr_target)
    
### Conversion of librispeech data for Intelligibilty
    # for flac_name in sorted(f.name for f in librispeech_dir.glob("*.flac")):
    #     # Get source file path
    #     source_path = librispeech_dir / flac_name
    #     wav_name = Path(flac_name).with_suffix(".wav")
    #     output_path = output_dir_I / wav_name
    #     # Extract features
    #     source_features = vc_model.get_features(source_path, mode=1)
    #     print(f"Extracted {source_features.shape[0]} features from {flac_name}")
    #     # Perform matching, vocode output and save
    #     output_features = vc_model.knn_matching(source_features, target_features)
    #     output_wav = vc_model.vocode(output_features[None].to(device)).cpu().squeeze()
    #     torchaudio.save(output_path, output_wav[None], vc_model.sr_target)
    #     print (f"Voice converted {flac_name} successfully")

### Conversion of librispeech data according to eval.csv for Similarity
    # output_dir_S.mkdir(parents=True, exist_ok=True)
    # print("Writing to:", output_dir_S)
    # with open(eval_csv) as f:
    #     for line in tqdm(f.readlines()):
    #         line = line.strip()
    #         if line[-1] == "0":
                
    #             # Set up filepath and source and target variables
    #             (source, target, source_key, _, _) = line.split(",")
    #             source_key_split = source_key.split("-")
    #             source_wav_fn = (
    #                 librispeech_dir_S
    #                 / source_key_split[0]
    #                 / source_key_split[1]
    #                 / source_key.split("/")[0]
    #             ).with_suffix(".flac")
    #             clip = source_key.split("/")[0]
    #             print(f"Converting speaker {source} clip: {clip} to speaker {target}")
                
    #             # Extract features for source
    #             print("Extracting source features...")
    #             source_features = vc_model.get_features(source_wav_fn, mode=1)
    #             print(f"Extracted {source_features.shape[0]} features from source speaker: {source}")
                
    #             # Load target features from .pt file
    #             print("Loading in target features...")
    #             filename_with_suffix = target + ".pt"
    #             target_fn = targets_dir / target / filename_with_suffix
    #             target_features = torch.load(target_fn)
    #             print(f"Loaded {target_features.shape[0]} features from target speaker: {target}")
                
    #             # Perform kNN matching to get output features
    #             print("Performing kNN matching...")
    #             output_features = vc_model.knn_matching(source_features, target_features)
                
    #             # Vocode and save the output
    #             print("Matching complete, vocoding and saving output...")
    #             cur_output_dir = Path(output_dir_S) / source_key.split("/")[0]
    #             cur_output_dir.mkdir(parents=True, exist_ok=True)
    #             output_fn = (cur_output_dir / source_key.split("/")[1]).with_suffix(
    #                 ".wav"
    #             )
    #             output_wav = vc_model.vocode(output_features[None].to(device)).cpu().squeeze()
    #             torchaudio.save(output_fn, output_wav[None], vc_model.sr_target)
    #             print("Succesfully converted")
                
### Timing end
    # end = time.time()
    # print(f"Finished in time: {(end - start)/60:.2f} minutes")
    
### Performance evaluation (intelligibility)
    # print("Evaluating intelligibility for target = Obama")
    # out = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/intelligibility/2_5_obama")
    # wer_mean1, wer_std1, cer_mean1, cer_std1 = evaluate_intelligibility(librispeech_dir, out)
    # print("Evaluating intelligibility for target = RFK")
    # out = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/intelligibility/2_5_rfk")
    # wer_mean2, wer_std2, cer_mean2, cer_std2 = evaluate_intelligibility(librispeech_dir, out)
    # wer_mean = (wer_mean1 + wer_mean2) / 2
    # wer_std = (wer_std1 + wer_std2) / 2
    # cer_mean = (cer_mean1 + cer_mean2) / 2
    # cer_std = (cer_std1 + cer_std2) / 2
    # print("Overall results:")
    # print(f"WER: {wer_mean:.2f}% +- {wer_std:.2f}%")
    # print(f"CER: {cer_mean:.2f}% +- {cer_std:.2f}%")
    
### Performance evaluation (similarity)
    # print("Evaluating similarity on librispeech data")
    # evaluate_similarity(librispeech_dir_S, output_dir_S, eval_csv)

if __name__ == "__main__":
    main()