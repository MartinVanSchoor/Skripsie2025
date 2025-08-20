import torch
import torch.nn.functional as F

def find_k_nearest_neighbors(src_feats, tgt_feats, k=5):
    """
    For each source feature, find k nearest target features
    using cosine similarity.
    
    Args:
        src_feats: (N_src, D) tensor
        tgt_feats: (N_tgt, D) tensor
        k: number of neighbors
    
    Returns:
        matched_tgts: (N_src, k, D) tensor of target features
    """
    src_norm = F.normalize(src_feats, dim=1)   # (N_src, D)
    tgt_norm = F.normalize(tgt_feats, dim=1)   # (N_tgt, D)

    sims = torch.matmul(src_norm, tgt_norm.T)  # (N_src, N_tgt)
    indices = torch.topk(sims, k=k, dim=1).indices  # (N_src, k)

    matched_tgts = tgt_feats[indices]  # (N_src, k, D)
    return matched_tgts


def learn_local_affine(X_src, X_tgt):
    """
    Learn affine mapping for one local neighborhood.
    X_src: (k, D), X_tgt: (k, D)
    Returns: W (D+1, D)
    """
    X_src_aug = torch.cat([X_src, torch.ones(X_src.size(0), 1, device=X_src.device)], dim=1)  # (k, D+1)
    result = torch.linalg.lstsq(X_src_aug, X_tgt)
    W = result.solution  # (D+1, D)
    return W


def apply_affine_mapping(X, W):
    """
    Apply affine transform W to X.
    X: (N, D)
    W: (D+1, D)
    Returns: (N, D)
    """
    X_aug = torch.cat([X, torch.ones(X.size(0), 1, device=X.device)], dim=1)
    return X_aug @ W


def align_features_via_local_affine(speakerB_feats, speakerA_feats, k=1):
    """
    Align Speaker B embeddings to Speaker A style using
    local affine mappings via nearest-neighbor neighborhoods.
    
    Args:
        speakerB_feats: (N_b, D) tensor
        speakerA_feats: (N_a, D) tensor
        k: number of neighbors to use per local affine
    
    Returns:
        B_feats_aligned: (N_b, D) tensor
    """
    N_b, D = speakerB_feats.shape
    matched_tgts = find_k_nearest_neighbors(speakerB_feats, speakerA_feats, k=k)  # (N_b, k, D)

    aligned = []
    for i in range(N_b):
        X_src = speakerB_feats[i].unsqueeze(0).repeat(k, 1)   # (k, D)
        X_tgt = matched_tgts[i]                               # (k, D)

        # Fit affine mapping locally
        W = learn_local_affine(X_src, X_tgt)

        # Apply mapping to the original feature
        aligned_feat = apply_affine_mapping(speakerB_feats[i].unsqueeze(0), W)
        aligned.append(aligned_feat)

    return torch.cat(aligned, dim=0)  # (N_b, D)
