"""Loss functions for the imbalance-aware training experiments (ELEC-444)."""
import numpy as np
import torch
import torch.nn as nn


def make_weighted_ce(train_labels, num_classes=7, scheme="inverse", device="cuda"):
    """Class-weighted cross-entropy.

    train_labels: 1-D array of TRAINING-set labels only.
    scheme: "inverse" = N / (K * n_c), "sqrt" = 1 / sqrt(n_c), normalized to mean 1.
    """
    counts = np.bincount(np.asarray(train_labels).ravel(), minlength=num_classes).astype(float)
    if scheme == "inverse":
        w = counts.sum() / (num_classes * counts)
    elif scheme == "sqrt":
        w = 1.0 / np.sqrt(counts)
        w = w / w.mean()
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    print(f"[{scheme}] class weights:", np.round(w, 2))
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
