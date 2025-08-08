import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

def get_cluster_centroids(features: torch.Tensor, n_clusters: int):
    """
    Cluster embeddings and return centroids as torch tensor.
    features: (N, D) CPU tensor or numpy array.
    Returns: (n_clusters, D) tensor
    """
    # KMeans expects numpy input on CPU
    X = features.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=features.dtype, device=features.device)
    return centroids

def match_clusters(centroids_B: torch.Tensor, centroids_A: torch.Tensor):
    """
    Match B's centroids to closest A centroids by cosine similarity.
    Returns indices of A matched to each B centroid.
    """
    B_norm = F.normalize(centroids_B, dim=1)
    A_norm = F.normalize(centroids_A, dim=1)

    sims = torch.matmul(B_norm, A_norm.T)  # (n_clusters_B, n_clusters_A)
    indices = torch.argmax(sims, dim=1)    # Best match in A for each B centroid
    return indices

def learn_affine_mapping(X_src, X_tgt):
    """
    Learn global affine mapping from X_src -> X_tgt with least squares.
    X_src: (N, D), X_tgt: (N, D)
    Returns: W (D+1, D)
    """
    X_src_aug = torch.cat([X_src, torch.ones(X_src.size(0), 1, device=X_src.device)], dim=1)
    result = torch.linalg.lstsq(X_src_aug, X_tgt)
    W = result.solution
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

def align_features_via_clusters(speakerB_feats, speakerA_feats, n_clusters=50):
    """
    Align Speaker B embeddings to Speaker A style using
    cluster centroid matching and a global affine transform.
    
    Args:
        speakerB_feats: (N_b, D) tensor
        speakerA_feats: (N_a, D) tensor
        n_clusters: number of clusters for k-means (tune as needed)
    
    Returns:
        B_feats_aligned: (N_b, D) tensor
    """
    # 1. Cluster features independently
    centroids_B = get_cluster_centroids(speakerB_feats, n_clusters)
    centroids_A = get_cluster_centroids(speakerA_feats, n_clusters)

    # 2. Match clusters B -> A by cosine similarity
    matched_indices = match_clusters(centroids_B, centroids_A)
    matched_centroids_A = centroids_A[matched_indices]

    # 3. Learn affine mapping from B centroids to matched A centroids
    W = learn_affine_mapping(centroids_B, matched_centroids_A)

    # 4. Apply affine mapping globally to all B features
    B_feats_aligned = apply_affine_mapping(speakerB_feats, W)

    return B_feats_aligned
