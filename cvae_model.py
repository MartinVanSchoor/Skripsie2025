"""
cvae_model.py

PyTorch implementation of a simple fully-connected conditional VAE (CVAE) for
generating speech features conditioned on a speaker identity vector.

Model design summary:
- Encoder: x (D_in) concatenated with c_proj -> MLP -> outputs mu & logvar (z_dim)
- Decoder: z concatenated with c_proj -> MLP -> reconstruct x (D_in)
- c_proj: small MLP projecting 512-d speaker id -> cond_dim (default 256)

Usage:
    from cvae_model import CVAE
    model = CVAE(x_dim=1024, c_dim=512, z_dim=128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple fully-connected MLP with ReLU activations and optional dropout."""
    def __init__(self, layer_sizes, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CVAE(nn.Module):
    def __init__(self,
                 x_dim=1024,
                 c_dim=512,
                 cond_proj_dim=256,
                 z_dim=128,
                 enc_hidden=[1024, 512],
                 dec_hidden=[512, 1024],
                 dropout=0.0):
        """
        Args:
            x_dim (int): dimension of input feature vector (e.g., 1024).
            c_dim (int): dimension of raw speaker identity vector (e.g., 512).
            cond_proj_dim (int): dimension of projected condition vector.
            z_dim (int): latent dimensionality.
            enc_hidden (list): hidden layer sizes for encoder (after concatenation).
            dec_hidden (list): hidden layer sizes for decoder (after concatenation).
            dropout (float): dropout probability.
        """
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.cond_proj_dim = cond_proj_dim
        self.z_dim = z_dim

        # Project condition vector c -> c_proj
        self.c_proj = MLP([c_dim, cond_proj_dim], dropout=dropout)

        # Encoder: input_dim = x_dim + cond_proj_dim
        enc_in = x_dim + cond_proj_dim
        enc_layers = [enc_in] + enc_hidden
        self.encoder_mlp = MLP(enc_layers + [enc_hidden[-1]], dropout=dropout)  # last ReLU included
        # linear heads for mu and logvar
        self.mu_layer = nn.Linear(enc_hidden[-1], z_dim)
        self.logvar_layer = nn.Linear(enc_hidden[-1], z_dim)

        # Decoder: input = z + cond_proj_dim
        dec_in = z_dim + cond_proj_dim
        dec_layers = [dec_in] + dec_hidden + [x_dim]
        self.decoder_mlp = MLP(dec_layers, dropout=dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def encode(self, x, c):
        """
        Args:
            x: (batch, x_dim)
            c: (batch, c_dim)
        Returns:
            mu, logvar each of shape (batch, z_dim)
        """
        c_proj = self.c_proj(c)  # (batch, cond_proj_dim)
        enc_in = torch.cat([x, c_proj], dim=-1)
        h = self.encoder_mlp(enc_in)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # use mean during eval by default

    def decode(self, z, c):
        """
        Args:
            z: (batch, z_dim)
            c: (batch, c_dim)
        Returns:
            recon_x: (batch, x_dim)
        """
        c_proj = self.c_proj(c)
        dec_in = torch.cat([z, c_proj], dim=-1)
        recon_x = self.decoder_mlp(dec_in)
        return recon_x

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

    def sample_prior(self, n_samples, c=None, device=None):
        """Sample from the prior p(z) and decode conditioned on c.
        If c is None, expects to be provided when calling decode (or repeats a provided c).
        """
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.z_dim, device=device)
        if c is None:
            raise ValueError("For conditional generation you must provide c (shape: [n_samples, c_dim])")
        return self.decode(z, c)

    def encode_batch(self, x_batch, c_batch):
        """Helper to return mu/logvar for a batch (no sampling)."""
        return self.encode(x_batch, c_batch)
