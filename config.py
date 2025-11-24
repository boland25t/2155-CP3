# config.py

CONFIG = {
    'latent_dim': 128,
    'hidden_dims': [512, 256, 128],
    'dropout_rate': 0.3,
    'use_residual': True,

    # training
    'learning_rate': 1e-3,
    'num_epochs': 500,
    'patience': 15,
    'beta_schedule': 'cosine',

    # data
    'batch_size': 64
}