# data.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import load_dataset_splits
from debug_utils import callout


def load_data(data_dir="dataset", batch_size=64):

    callout("Loading dataset splits...")
    splits = load_dataset_splits(data_dir)

    callout("Extracting arrays from splits...")
    X_train, mask_train = splits['train']['original'], splits['train']['missing_mask']
    X_val,   mask_val   = splits['val']['original'],   splits['val']['missing_mask']
    X_test,  mask_test  = splits['test']['original'],  splits['test']['missing_mask']
    X_test2, mask_test2 = splits['test2']['imputed'],  splits['test2']['missing_mask']

    callout("Creating TensorDatasets...")
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor((~mask_train).astype(float)))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor((~mask_val).astype(float)))
    test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.FloatTensor((~mask_test).astype(float)))
    test2_ds = TensorDataset(torch.FloatTensor(X_test2), torch.FloatTensor((~mask_test2).astype(float)))

    callout("Building DataLoaders...")
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val':   DataLoader(val_ds,   batch_size=batch_size),
        'test':  DataLoader(test_ds,  batch_size=batch_size),
        'test2': DataLoader(test2_ds, batch_size=batch_size),
    }
    callout("Data loading complete.")