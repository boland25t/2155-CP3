# model_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from debug_utils import conditional_callout


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, use_residual=True, dropout_rate=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_residual = use_residual

        self.feature_importance = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 2, input_dim),
            nn.Sigmoid()
        )

        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim * 2
        for h in hidden_dims:
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = h

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        self.decoder_layers = nn.ModuleList()
        prev_dim = latent_dim + input_dim + input_dim
        for h in reversed(hidden_dims):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            prev_dim = h

        self.output_layer = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x, mask):
        conditional_callout("Encoding batch...", freq=50)
        mask_f = mask.float()
        importance = self.feature_importance(torch.cat([x * mask_f, mask_f], dim=1))
        weighted = x * mask_f * importance
        h = torch.cat([weighted, mask_f], dim=1)

        for i, layer in enumerate(self.encoder_layers):
            prev = h
            h = layer(h)
            if self.use_residual and i > 0 and prev.shape == h.shape:
                h = h + prev

        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        conditional_callout("Sampling z...", freq=80)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_obs, mask):
        conditional_callout("Decoding...", freq=50)
        mask_f = mask.float()
        x_masked = x_obs * mask_f

        pos = torch.arange(self.input_dim, device=z.device).float()
        pos = pos.unsqueeze(0).expand(z.size(0), -1) / self.input_dim

        h = torch.cat([z, x_masked, pos], dim=1)
        for layer in self.decoder_layers:
            h = layer(h)

        return self.output_layer(h)

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x, mask), mu, logvar
