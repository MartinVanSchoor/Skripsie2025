import pandas as pd
import numpy as np

# Load your eval.csv
csv_path = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/eval.csv"   # change this to your actual path
eval_df = pd.read_csv(csv_path)

# Split into conversions (label=0) and ground truth (label=1)
conversions = eval_df[eval_df["label"] == 0].copy()
ground_truth = eval_df[eval_df["label"] == 1].copy()

# --- Trim conversions: keep only 5 target speakers per source speaker ---
trimmed_conversions = []
for spk in conversions["src_speaker"].unique():
    spk_df = conversions[conversions["src_speaker"] == spk]
    tgt_candidates = spk_df["tgt_speaker"].unique()
    
    # Pick 5 random target speakers
    chosen_tgts = np.random.choice(tgt_candidates, size=5, replace=False)
    trimmed_spk_df = spk_df[spk_df["tgt_speaker"].isin(chosen_tgts)]
    
    trimmed_conversions.append(trimmed_spk_df)

trimmed_conversions = pd.concat(trimmed_conversions, ignore_index=True)

# --- Trim ground truth: keep the first 25 pairs per speaker ---
trimmed_ground_truth = []
for spk in ground_truth["src_speaker"].unique():
    spk_df = ground_truth[ground_truth["src_speaker"] == spk]
    spk_df = spk_df.head(25)  # take first 25
    trimmed_ground_truth.append(spk_df)

trimmed_ground_truth = pd.concat(trimmed_ground_truth, ignore_index=True)

# --- Combine back ---
trimmed_eval = pd.concat([trimmed_conversions, trimmed_ground_truth], ignore_index=True)

# Save to new file
out_path = "/mnt/c/Users/marti/Documents/Werk/Universiteit/Skripsie/Skripsie2025/data/eval_trimmed_2.csv"
trimmed_eval.to_csv(out_path, index=False)

print(f"Trimmed dataset saved to {out_path}")
print(f"Total rows: {len(trimmed_eval)}")
print(f"Conversions: {len(trimmed_conversions)}, Ground truth: {len(trimmed_ground_truth)}")
