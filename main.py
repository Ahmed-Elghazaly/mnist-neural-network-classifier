#!/usr/bin/env python3


# -- Imports ------------------------------------------------------------------
import copy
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# -- Reproducibility ----------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Section 1: Data Preparation
# Ref: [analysis.md: Data Preparation](analysis.md#section-1-data-preparation)
# =============================================================================
print("\n" + "=" * 70)
print("Section 1: Data Preparation")
print("=" * 70)

# Download MNIST via torchvision (equivalent to Kaggle MNIST)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
full_test_orig = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Combine train+test into one pool so we can do our own 60/20/20 split
all_data = torch.cat([full_train.data, full_test_orig.data], dim=0).float()
all_labels = torch.cat([full_train.targets, full_test_orig.targets], dim=0)

# Normalize manually (same stats)
all_data = (all_data / 255.0 - 0.1307) / 0.3081
all_data = all_data.unsqueeze(1)  # (N, 1, 28, 28)

print(f"Total samples : {len(all_data)}")
print(f"Image shape   : {all_data.shape[1:]}")
print(f"Classes       : {torch.unique(all_labels).tolist()}")

# -- Stratified split: 60% train, 20% val, 20% test --------------------------
indices = np.arange(len(all_data))
labels_np = all_labels.numpy()

# Ref: [analysis.md: Stratified Split](analysis.md#section-1-data-preparation)
idx_train, idx_temp, y_train_s, y_temp = train_test_split(
    indices, labels_np, test_size=0.40, stratify=labels_np, random_state=SEED
)
idx_val, idx_test, _, _ = train_test_split(
    idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

X_train, y_train = all_data[idx_train], all_labels[idx_train]
X_val, y_val = all_data[idx_val], all_labels[idx_val]
X_test, y_test = all_data[idx_test], all_labels[idx_test]

print(f"Train : {len(X_train)}  |  Val : {len(X_val)}  |  Test : {len(X_test)}")


# -- DataLoader factory -------------------------------------------------------
# Ref: [analysis.md: DataLoader Factory](analysis.md#section-1-data-preparation)
def make_loaders(batch_size, x_tr=X_train, y_tr=y_train,
                 x_va=X_val, y_va=y_val, x_te=X_test, y_te=y_test):
    """Return (train_loader, val_loader, test_loader) for given batch_size."""
    train_ds = TensorDataset(x_tr, y_tr)
    val_ds = TensorDataset(x_va, y_va)
    test_ds = TensorDataset(x_te, y_te)
    kw = dict(pin_memory=True, num_workers=0)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=512, shuffle=False, **kw),
            DataLoader(test_ds, batch_size=512, shuffle=False, **kw))


# =============================================================================
# Section 2: Neural Network Architectures
# =============================================================================

# -- 2a. Feed-Forward Network (FNN) ------------------------------------------
# Ref: [analysis.md: FNN Architecture](analysis.md#section-2-neural-network-architectures)
class FeedForwardNet(nn.Module):
    """Configurable feed-forward network for MNIST (28x28 -> 10)."""

    def __init__(self, hidden_sizes=(256, 128), dropout_p=0.0):
        super().__init__()
        layers = []
        in_features = 28 * 28
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_features = h
        layers.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# -- 2b. Convolutional Network (Bonus) ---------------------------------------
# Ref: [analysis.md: CNN Architecture](analysis.md#section-2-neural-network-architectures)
class ConvNet(nn.Module):
    """
    CNN for MNIST with optional Dropout and LayerNorm (bonus).
    Architecture: Conv->LN->ReLU->Conv->LN->ReLU->Pool->FC->FC
    """

    def __init__(self, use_dropout=True, use_layernorm=True):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_layernorm = use_layernorm

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28->28x28
        self.ln1 = nn.LayerNorm([32, 28, 28]) if use_layernorm else nn.Identity()

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28->28x28
        self.ln2 = nn.LayerNorm([64, 28, 28]) if use_layernorm else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)  # ->14x14

        # Conv block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 14x14->14x14
        self.ln3 = nn.LayerNorm([128, 14, 14]) if use_layernorm else nn.Identity()

        self.pool2 = nn.MaxPool2d(2, 2)  # ->7x7

        self.drop1 = nn.Dropout(0.25) if use_dropout else nn.Identity()
        self.drop2 = nn.Dropout(0.50) if use_dropout else nn.Identity()

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.ln_fc = nn.LayerNorm(256) if use_layernorm else nn.Identity()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.ln3(self.conv3(x)))
        x = self.pool2(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.ln_fc(self.fc1(x)))
        x = self.drop2(x)
        x = self.fc2(x)
        return x


