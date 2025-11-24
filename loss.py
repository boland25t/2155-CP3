# loss.py

import torch
import torch.nn.functional as F
import numpy as np

def vae_loss(recon, x, mu, logvar, mask, beta=1.0):
    mask_f = mask.float()
    diff = (recon - x)**2 * mask_f
    recon_loss = diff.sum() / (mask_f.sum() + 1e-8)

    std_recon = F.mse_loss(recon * mask_f, x * mask_f)
    recon_loss = 0.7 * recon_loss + 0.3 * std_recon

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl /= x.size(0)

    return recon_loss + beta * kl, recon_loss, kl

def beta_schedule(epoch, total, mode="cosine"):
    if mode == "cosine":
        return 0.5 * (1 + np.cos(np.pi * (1 - epoch / total)))
    return 1.0
