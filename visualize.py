# visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def savefig(name):
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"📁 Saved plot: {path}")


def plot_mask_heatmap(mask, title, fname):
    plt.figure(figsize=(10, 4))
    sns.heatmap(mask[:200], cbar=False)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Samples")
    savefig(fname)


def plot_feature_distribution(idx, name, originals, imputations, mask):
    missing = (mask[:, idx] == 0)
    imp_vals = imputations[missing, idx]
    true_vals = originals[missing, idx]

    plt.figure(figsize=(8,4))
    sns.kdeplot(true_vals, label="True", fill=True)
    sns.kdeplot(imp_vals, label="Imputed", fill=True)
    plt.legend()
    plt.title(f"{name}: Feature {idx}")
    savefig(f"distribution_feature_{idx}.png")


def plot_feature_importance(model):
    W = model.feature_importance[2].weight.detach().cpu().numpy()
    plt.figure(figsize=(10,8))
    sns.heatmap(W, cmap='coolwarm')
    plt.title("VAE Learned Feature Importance")
    savefig("feature_importance.png")


def plot_correlation_comparison(true_data, imputed_data, mask):
    missing = (mask == 0)

    # Extract missing-only true & imputed values
    true = true_data[missing]
    imp = imputed_data[missing]

    # Must reshape if flattened
    if true.ndim == 1:
        print("⚠️ Not enough missing data to compute correlation (true). Skipping.")
        return
    if imp.ndim == 1:
        print("⚠️ Not enough missing data to compute correlation (imputed). Skipping.")
        return

    # Must have at least 2 samples to compute correlation
    if true.shape[0] < 2 or imp.shape[0] < 2:
        print(f"⚠️ Too few missing samples ({true.shape[0]}) to compute correlation. Skipping.")
        return

    # Must have at least 2 features
    if true.shape[1] < 2 or imp.shape[1] < 2:
        print(f"⚠️ Too few features ({true.shape[1]}) to compute correlation. Skipping.")
        return

    # Compute correlations
    corr_true = np.corrcoef(true.T)
    corr_imp = np.corrcoef(imp.T)

    # Validate matrix shapes
    if corr_true.ndim != 2 or corr_imp.ndim != 2:
        print("⚠️ Correlation output invalid. Probably too few values. Skipping.")
        return

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    sns.heatmap(corr_true, center=0, cmap='coolwarm')
    plt.title("True Correlation")

    plt.subplot(1,2,2)
    sns.heatmap(corr_imp, center=0, cmap='coolwarm')
    plt.title("Imputed Correlation")

    savefig("correlation_comparison.png")
