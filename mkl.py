import numpy as np
import torch, torchaudio
import scipy.io.wavfile as wavf

# No. of dimensions of WavLM features
EMBED_LEN = 1024

def MKL(A, B):
    EPS = 2.2204e-16
    
    Da2, Ua = np.linalg.eig(A)
    Da2 = np.diag(Da2)
    Da2[Da2 < 0] = 0
    Da = np.sqrt(Da2 + EPS)
    C = Da @ np.transpose(Ua) @ B @ Ua @ Da
    
    Dc2, Uc = np.linalg.eig(C)
    Dc2 = np.diag(Dc2)
    Dc2[Dc2 < 0] = 0
    Dc = np.sqrt(Dc2 + EPS)
    Da_inv = np.diag(1 / (np.diag(Da)))
    T = Ua @ Da_inv @ Uc @ Dc @ np.transpose(Uc) @ Da_inv @ np.transpose(Ua)
    return T


def apply_mkl(X0, X1):
    A = np.cov(X0, rowvar=False)
    B = np.cov(X1, rowvar=False)
    T = MKL(A, B)
    mX0 = np.mean(X0, axis=0)
    mX1 = np.mean(X1, axis=0)
    XR = (X0 - mX0) @ T + mX1
    XR = np.real(XR)
    return XR


def apply_mkl_batched(X0, X1, batch_size):
    XR = np.zeros_like(X0)
    for i in range(0, EMBED_LEN, batch_size):
        if i + batch_size < EMBED_LEN:
            XR[:,i:i+batch_size] = apply_mkl(X0[:,i:i+batch_size], X1[:,i:i+batch_size])
        elif i < EMBED_LEN - 1:
            XR[:,i:] = apply_mkl(X0[:,i:], X1[:,i:])
        elif i == EMBED_LEN - 1:
            XR[:,i] = X0[:,i]
    return XR
