import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=64, num_speakers=51):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.spk_embedding = nn.Embedding(num_speakers, 64)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x, speaker_ids):
        spk_embed = self.spk_embedding(speaker_ids)
        x = torch.cat([x, spk_embed], dim=1)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, speaker_ids):
        spk_embed = self.spk_embedding(speaker_ids)
        z = torch.cat([z, spk_embed], dim=1)
        return self.decoder(z)

    def forward(self, x, speaker_ids):
        mu, logvar = self.encode(x, speaker_ids)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, speaker_ids)
        return x_recon, mu, logvar
