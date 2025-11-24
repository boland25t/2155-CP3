# train.py

import torch
from tqdm import tqdm
from loss import vae_loss, beta_schedule
from debug_utils import callout, conditional_callout


def evaluate(model, loader, device, beta):
    model.eval()
    total = 0
    with torch.no_grad():
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            recon, mu, logvar = model(x, mask)
            loss, _, _ = vae_loss(recon, x, mu, logvar, mask, beta)
            total += loss.item()
    return total / len(loader)


def train(model, loaders, config, device):

    callout("Training initialized.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2)

    best, patience = float("inf"), 0

    for epoch in range(config['num_epochs']):
        callout(f"Epoch {epoch+1}/{config['num_epochs']} started...")

        model.train()
        beta = beta_schedule(epoch, config['num_epochs'], config['beta_schedule'])
        train_loss = 0

        for batch_idx, (x, mask) in enumerate(tqdm(loaders['train'], desc=f"Epoch {epoch+1}")):
            conditional_callout("Forward pass...", freq=400)

            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x, mask)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, mask, beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        val_loss = evaluate(model, loaders['val'], device, beta)
        scheduler.step()

        callout(f"Summary: train={train_loss:.4f} | val={val_loss:.4f} | beta={beta:.3f}")

        if val_loss < best:
            callout("New best model saved.")
            best = val_loss
            patience = 0
            torch.save(model.state_dict(), "best_vae.pth")
        else:
            patience += 1
            if patience >= config['patience']:
                callout("Early stopping.")
                break

    model.load_state_dict(torch.load("best_vae.pth"))
    return model
