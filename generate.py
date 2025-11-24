# generate.py

import torch
import numpy as np
from tqdm import tqdm
from debug_utils import callout


def generate_samples(model, loader, device, n=100):

    callout("Beginning test2 sample generation...")
    model.eval()
    all_samples = []

    with torch.no_grad():
        for batch_data, batch_mask in tqdm(loader, desc="Generating"):
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)

            bs, nf = batch_data.size(0), batch_data.size(1)
            samples = np.zeros((bs, n, nf))

            for j in range(n):
                recon, mu, logvar = model(batch_data, batch_mask)
                mask_f = batch_mask.float()
                imputed = batch_data * mask_f + recon * (1 - mask_f)
                samples[:, j, :] = imputed.cpu().numpy()

            all_samples.append(samples)

    callout("Finished generating all samples.")
    return np.vstack(all_samples)
