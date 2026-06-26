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


# =============================================================================
# Section 3: Training Loop
# =============================================================================

# Ref: [analysis.md: Training Loop](analysis.md#section-3-training-loop)
def train_model(model, train_loader, val_loader, lr=0.01, epochs=20,
                device=DEVICE, verbose=True):
    """
    Custom training loop with SGD + CrossEntropyLoss.
    Returns dict with per-epoch metrics.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # -- Training phase -----------------------------------------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # -- Validation phase ---------------------------------------------
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += X_batch.size(0)

        val_loss = running_loss / total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    return history


def evaluate_model(model, loader, device=DEVICE):
    """Return (accuracy, all_preds, all_targets)."""
    model.eval()
    all_preds, all_targets = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            preds = logits.argmax(1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_targets)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def plot_curves(history, title="", save_path=None):
    """Plot training/validation loss and accuracy side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch");
    ax1.set_ylabel("Loss");
    ax1.set_title(f"Loss - {title}")
    ax1.legend();
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch");
    ax2.set_ylabel("Accuracy");
    ax2.set_title(f"Accuracy - {title}")
    ax2.legend();
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel("Predicted");
    ax.set_ylabel("True");
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results_dict, metric, ylabel, title, save_path=None):
    """Overlay curves from multiple experiments."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, hist in results_dict.items():
        epochs = range(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], "-o", markersize=2, label=str(label))
    ax.set_xlabel("Epoch");
    ax.set_ylabel(ylabel);
    ax.set_title(title)
    ax.legend();
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Section 4: Baseline Training
# Ref: [analysis.md: Baseline Training](analysis.md#section-4-baseline-training)
# =============================================================================
print("\n" + "=" * 70)
print("Section 4: Baseline Training (lr=0.01, bs=64, hidden=[256,128])")
print("=" * 70)

EPOCHS = 25
BS_DEFAULT = 64
LR_DEFAULT = 0.01

train_loader, val_loader, test_loader = make_loaders(BS_DEFAULT)

baseline_model = FeedForwardNet(hidden_sizes=(256, 128))
baseline_hist = train_model(baseline_model, train_loader, val_loader,
                            lr=LR_DEFAULT, epochs=EPOCHS)

plot_curves(baseline_hist, "Baseline FNN (256-128)",
            os.path.join(OUTPUT_DIR, "baseline_curves.png"))

baseline_acc, preds_bl, targets_bl = evaluate_model(baseline_model, test_loader)
print(f"\n  * Baseline Test Accuracy: {baseline_acc:.4f}")

plot_confusion(targets_bl, preds_bl, "Baseline Confusion Matrix",
               os.path.join(OUTPUT_DIR, "baseline_confusion.png"))

# =============================================================================
