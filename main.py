# main.py

import torch
import numpy as np

from config import CONFIG
from data import load_data
from model_vae import VAE
from train import train
from generate import generate_samples
from visualize import *
from debug_utils import callout


def main():

    callout("Starting CP3 pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = load_data(batch_size=CONFIG['batch_size'])
    input_dim = loaders['train'].dataset[0][0].shape[0]

    # === MASK VISUALIZATIONS ===
    train_mask = loaders['train'].dataset.tensors[1].numpy()
    val_mask   = loaders['val'].dataset.tensors[1].numpy()
    test_mask  = loaders['test'].dataset.tensors[1].numpy()

    plot_mask_heatmap(train_mask, "Train Mask Heatmap", "mask_train.png")
    plot_mask_heatmap(val_mask,   "Val Mask Heatmap",   "mask_val.png")
    plot_mask_heatmap(test_mask,  "Test Mask Heatmap",  "mask_test.png")

    # === MODEL INITIALIZATION ===
    model = VAE(
        input_dim=input_dim,
        latent_dim=CONFIG['latent_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        dropout_rate=CONFIG['dropout_rate'],
        use_residual=CONFIG['use_residual']
    ).to(device)

    model = train(model, loaders, CONFIG, device)

    # === FEATURE IMPORTANCE PLOT ===
    plot_feature_importance(model)

    # === GENERATE SAMPLES FOR TEST SET FOR VISUALS ===
    callout("Generating evaluation samples for visualization...")
    test_samples = generate_samples(model, loaders['test'], device, n=1)
    test_samples = test_samples[:,0,:]  # shape (N, F)

    # Extract original + mask
    test_data = loaders['test'].dataset.tensors[0].numpy()
    test_mask_np = loaders['test'].dataset.tensors[1].numpy()

    # === DISTRIBUTION PLOTS ===
    for idx in [0, 5, 10, 15, 20]:
        plot_feature_distribution(idx, f"Feature_{idx}",
                                  test_data, test_samples,
                                  test_mask_np)

    # === CORRELATION COMPARISON ===
    plot_correlation_comparison(test_data, test_samples, test_mask_np)

    # === TEST2 SUBMISSION ===
    callout("Generating final test2 samples...")
    test2_samples = generate_samples(model, loaders['test2'], device, n=100)

    fid = np.random.randint(1e8, 9e8)
    np.save(f"{fid}.npy", test2_samples)
    callout(f"Saved: {fid}.npy")


if __name__ == "__main__":
    main()
