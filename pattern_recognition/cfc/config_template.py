labels = {
    'sin+cos': 0,
    'sin+tan': 1,
    'tan+cos': 2,
}

model_params = {
    'in_features': 2,
    'out_features': len(labels),
    'units': 32,
    'max_seq_len': 67,
}

train_params = {
    'batch_size': 32,
    'max_epochs': 400,
    'base_lr': 0.02,
    'decay_lr': 0.97,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'early_stop_patience': 40,
    'log_interval': 5
}
