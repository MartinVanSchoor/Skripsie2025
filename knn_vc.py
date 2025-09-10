import numpy as np
import torch
from torch import Tensor
import torchaudio
from sklearn.neighbors import NearestNeighbors
import torchaudio.functional as F
import time
import intelligibility
import similarity
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from mkl import apply_mkl_batched

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
    eer_mean, eer_std = similarity.speaker_similarity(args)
    return eer_mean, eer_std

class kNN_VC(torch.nn.Module):
    def __init__(self, wavlm, hifigan, k, device, batch_size):
        super().__init__()
        self.wavlm = wavlm.eval()
        self.hifigan = hifigan.eval()
        self.k = k
        self.device = device 
        self.sr_target = 16000
        self.batch_size = batch_size
        
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
    
    @torch.inference_mode()
    def expand_feature_space(self, target_id, target_features, train_dir, train_ids, train_speakers, batch_size):
        """ 
        Expands the source speaker's feature space by sampling from
        the most similar speaker in the librispeech train-clean set,
        and interpolating in feature space.
        """
        # Find most similar speaker and retrieve features
        closest_speaker = kNN_VC.find_most_similar_speaker(target_id, train_ids, train_speakers)
        print(f"Closest speaker is: {closest_speaker}, loading features...")
        feat_dir = train_dir / closest_speaker / f"{closest_speaker}.npy"
        train_features = np.load(feat_dir) 
        train_features = torch.from_numpy(train_features).to("cpu")

        # Retrieve the necessary amount of features and convert to appropriate device
        #kNN approach
        diff = 8996 - target_features.shape[0]
        train_features = train_features[:diff].to(self.device) 
        print(f"Loaded {train_features.shape[0]} features from speaker {closest_speaker}")
        #MKL approach
        # diff = 512 - target_features.shape[0]
        # train_features = train_features[:diff].to(self.device) 
        # print(f"Loaded {train_features.shape[0]} features from speaker {closest_speaker}")

        # Apply MKl transport mapping to make the training features more similar to the target features
        train_features = self.perform_mkl_mapping(train_features, target_features)

        # Concatenate
        expanded_features = torch.cat([target_features, train_features], dim=0)

        return expanded_features
    
    @torch.inference_mode()
    def find_most_similar_speaker(x: torch.Tensor, y: torch.Tensor, speakers):
        """ 
        Finds the most similar speaker to x in the train-clean set 
        using fast cosine similarity 
        x: [D]           — target speaker id
        y: [N, D]        — train-clean speaker id's
        """
        # Normalize input vectors
        x_norm = F.normalize(x, dim=0)        
        y_norm = F.normalize(y, dim=1)         
        
        # Compute cosine similarities
        similarities = torch.matmul(y_norm, x_norm)  

        # Find most similar speaker and return
        _, best_idx = torch.max(similarities, dim=0)
        return speakers[best_idx.item()]
    
    @torch.inference_mode()
    def perform_mkl_mapping(self, source_features, target_features):
        """ 
        Performs Monge-Kantorovich linear transport mapping and returns the result
        """
        #Sort dimensions by standard devation
        idxes = torch.argsort(source_features.std(0), descending=True)
        X0 = source_features[:, idxes].cpu().numpy()
        X1 = target_features[:, idxes].cpu().numpy()

        #Ensure batch size isn't too large, and obtain and apply mapping
        self.batch_size = min(len(X1)-4, self.batch_size)
        XR = apply_mkl_batched(X0, X1, self.batch_size)
        output_features = source_features.clone()
        output_features[:, idxes] = torch.Tensor(XR).to(self.device)
        return output_features
        
def main(target_length, set, k, batch_size):
    
    print(f"Using {target_length} secs of target audio for the {set} set")
### Specify filenames and other variables
 ## For Laptop
    # device = "cpu"
    # perf = "/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/performance/08-05-2025.txt"
    # eval_csv = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/eval.csv")
    # librispeech_dir = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/librispeech/Librispeech/dev-clean")
    # targets_dir = Path(f"/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/librispeech_targets/{target_length}")
    # output_dir = Path(f"/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/converted/{target_length}")
    # train_dir = Path("/mnt/c/Users/marti/Tuts_Projects/Skripsie/Skripsie2025/data/train")
 ## For Desktop
    device = "cuda"
    perf = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/performance/red_csv_vanilla.txt"
    eval_csv = Path(f"/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/eval.csv")
    librispeech_dir = Path(f"/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/librispeech/LibriSpeech/dev-clean")
    targets_dir = Path(f"/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/librispeech_targets/dev/{target_length}")
    output_dir = Path(f"/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/converted/dev/{target_length}")
    # train_dir = Path("/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/train_100")
    train_dir = Path("/mnt/d/librispeech_train/train")
    k_top = k
    
### Load in the neccessary models {SSL feature extractor (WavLM) and Vocoder (HiFi-GAN)}
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    # Load speaker id's for sampling
    if (target_length < 180):
        ids = torch.empty(0, 512).to(device)
        speakers = np.array([], dtype=str)
        for speaker_dir in tqdm(sorted(train_dir.iterdir()), desc="Loading training id's"):
            speaker_name = speaker_dir.name
            speaker_id_fn = speaker_dir / f"{speaker_name}_id.npy"
            id = np.load(speaker_id_fn)
            id = torch.from_numpy(id).unsqueeze(0).to(device)
            speakers = np.append(speakers, speaker_name)
            ids = torch.cat([ids, id], dim=0)
    
### Timing start and model initialization
    start = time.time()
    vc_model = kNN_VC(wavlm, hifigan, k_top, device, batch_size)

### Conversion of librispeech dev-clean set data according to eval.csv 
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Writing to:", output_dir)
    with open(eval_csv) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line[-1] == "0":

                # Set up filepath and source and target variables
                (source, target, source_key, target_key, _) = line.split(",")
                #Source filename
                source_key_split = source_key.split("-")
                source_wav_fn = (
                    librispeech_dir
                    / source_key_split[0]
                    / source_key_split[1]
                    / source_key.split("/")[0]
                ).with_suffix(".flac")
                clip = source_key.split("/")[0]
                #Output filename
                cur_output_dir = Path(output_dir) / source_key.split("/")[0]
                cur_output_dir.mkdir(parents=True, exist_ok=True)
                output_fn = (cur_output_dir / source_key.split("/")[1]).with_suffix(
                    ".wav"
                )
                #Target filename
                target_wav_fn = (librispeech_dir / target_key).with_suffix(".flac")
                #Skip conversions if neccessary
                # if output_fn.exists():
                #     continue
                print(f"Converting speaker {source} clip: {clip} to speaker {target}")
                
                # Extract features for source
                print("Extracting source features...")
                source_features = vc_model.get_features(source_wav_fn, mode=1)
                print(f"Extracted {source_features.shape[0]} features from source speaker: {source}")
                
                # Load target features from .pt file
                print("Loading in target features...")
                feat_fn_with_suffix = target + ".pt"
                id_fn_with_suffix = target + "_id.pt"
                target_fn = targets_dir / target / feat_fn_with_suffix
                target_id_fn = targets_dir / target / id_fn_with_suffix
                target_features = torch.load(target_fn)
                target_id = torch.load(target_id_fn)
                target_features = target_features.to(device)
                target_id = target_id.to(device)
                print(f"Loaded {target_features.shape[0]} features from target speaker: {target}")
                print(f"Loaded target id with {target_id.shape[0]} dimensions")

                # # Extract target features from target utterance
                # print("Extracting source features...")
                # target_features = vc_model.get_features(target_wav_fn, mode=1)
                # print(f"Extracted {target_features.shape[0]} features from target speaker: {target}")
    
                # If the target features are too few, expand the feature space
                if (target_length < 180): 
                    print("Insuficcient target data, expanding target set...")
                    target_features = vc_model.expand_feature_space(target_id, target_features, train_dir, ids, speakers, batch_size)
                    print(f"New target set has {target_features.shape[0]} features")

                # Perform kNN matching 
                print(f"Performing kNN matching...")
                output_features = vc_model.knn_matching(source_features, target_features)

                # Perform MKL mapping
                # print("Performing MKL mapping")
                # output_features = vc_model.perform_mkl_mapping(source_features, target_features)
                
                # Vocode and save the output
                print("Matching complete, vocoding and saving output...")
                output_wav = vc_model.vocode(output_features[None].to(device)).cpu().squeeze()
                torchaudio.save(output_fn, output_wav[None], vc_model.sr_target)
                print("Succesfully converted")
                torch.cuda.empty_cache()   

### Timing end
    end = time.time()
    print(f"Finished all conversions in time: {(end - start)/60:.2f} minutes")
    
### Performance evaluation
    print("Evaluating similarity")
    eer_mean, eer_std = evaluate_similarity(librispeech_dir, output_dir, eval_csv)
    print("Evaluating intelligibility")
    wer_mean, wer_std, cer_mean, cer_std = evaluate_intelligibility(librispeech_dir, output_dir)
    with open(perf, "a") as f:
        f.write(f"The performance of the kNN_VC model for {target_length} seconds of target audio and K={batch_size} from the {set}-clean set is:\n")
        f.write("Intelligiblity:\n")
        f.write(f"WER: {wer_mean} +- {wer_std}\n")
        f.write(f"CER: {cer_mean} +- {cer_std}\n")
        f.write("Similarity:\n")
        f.write(f"EER: {eer_mean} +- {eer_std}\n")
        f.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_length', type=int, default=3, help='Target length for features')
    parser.add_argument('--set', type=str, default="dev", help='Librispeech set to use (dev or test)')
    parser.add_argument('--k', type=int, default=4, help='k for kNN')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (K) for MKL mapping (how many dimensions per group)')
    args = parser.parse_args()
    main(args.target_length, args.set, args.k, args.batch_size)