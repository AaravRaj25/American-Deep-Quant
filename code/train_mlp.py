"""
train_mlp.py
Week 6, Part B: train a PyTorch MLP to regress American put price from
(S0, K, T, r, sigma), using the CRR-labeled dataset from generate_dataset.py.

- 80/10/10 train/val/test split (fixed seed)
- Standardization fit on TRAINING split only (no leakage)
- MLP with 3 hidden layers
- Logs train/val MSE per epoch, saves best-val checkpoint (not just final)
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
DATA_PATH = "/home/claude/week6/data/american_put_dataset.csv"
CKPT_PATH = "/home/claude/week6/checkpoints/best_mlp.pt"
SCALER_PATH = "/home/claude/week6/checkpoints/scaler.json"
HISTORY_PATH = "/home/claude/week6/checkpoints/history.json"
SPLIT_PATH = "/home/claude/week6/data/test_split.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)

FEATURES = ["S0", "K", "T", "r", "sigma"]
TARGET = "price"


class PutPricerMLP(nn.Module):
    """3-hidden-layer MLP: 5 -> 64 -> 64 -> 32 -> 1"""
    def __init__(self, in_dim=5, hidden=(64, 64, 32)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_data(df, seed=SEED):
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return df.iloc[train_idx].reset_index(drop=True), \
           df.iloc[val_idx].reset_index(drop=True), \
           df.iloc[test_idx].reset_index(drop=True)


def main():
    df = pd.read_csv(DATA_PATH)
    train_df, val_df, test_df = split_data(df)
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # save test split for the evaluation script (reproducibility, no leakage)
    test_df.to_csv(SPLIT_PATH, index=False)

    # Standardize using TRAIN stats only
    mu = train_df[FEATURES].mean()
    sd = train_df[FEATURES].std()
    scaler = {"mean": mu.to_dict(), "std": sd.to_dict()}
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler, f, indent=2)

    def to_tensor(d):
        X = ((d[FEATURES] - mu) / sd).values.astype(np.float32)
        y = d[TARGET].values.astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

    X_train, y_train = to_tensor(train_df)
    X_val, y_val = to_tensor(val_df)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128,
                               shuffle=True, generator=torch.Generator().manual_seed(SEED))

    model = PutPricerMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 200
    best_val = float("inf")
    history = {"train_mse": [], "val_mse": []}
    patience, bad_epochs = 25, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_mse = running / len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = loss_fn(val_pred, y_val).item()

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_mse": val_mse}, CKPT_PATH)
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE {train_mse:8.4f} | val MSE {val_mse:8.4f} "
                  f"| best val MSE {best_val:8.4f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    ckpt = torch.load(CKPT_PATH, weights_only=False)
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val MSE {ckpt['val_mse']:.4f} "
          f"(val RMSE {ckpt['val_mse']**0.5:.4f})")


if __name__ == "__main__":
    main()
