"""
evaluate.py
Week 6, Part C: evaluation and finance sanity checks.

Produces:
  - test MAE / RMSE / max abs error
  - MAE by moneyness bucket (deep ITM / near ATM / deep OTM, in put terms)
  - predicted-vs-binomial scatter plot
  - NN price surface vs binomial surface for fixed K=100, r=5%, sigma=25%
  - sanity checks: non-negativity, monotonic decreasing in S0, intrinsic-value floor
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/claude/week6/code")
from train_mlp import PutPricerMLP, FEATURES, TARGET
from crr_pricer import crr_put_price

DATA_DIR = "/home/claude/week6/data"
CKPT_DIR = "/home/claude/week6/checkpoints"
PLOT_DIR = "/home/claude/week6/plots"

with open(f"{CKPT_DIR}/scaler.json") as f:
    scaler = json.load(f)
mu = pd.Series(scaler["mean"])
sd = pd.Series(scaler["std"])

model = PutPricerMLP()
ckpt = torch.load(f"{CKPT_DIR}/best_mlp.pt", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val MSE {ckpt['val_mse']:.4f})")


def predict(df):
    X = ((df[FEATURES] - mu) / sd).values.astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    return pred


def predict_raw(S0, K, T, r, sigma):
    """Predict from raw numpy arrays (used for the surface comparison)."""
    df = pd.DataFrame({"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma})
    return predict(df)


# ---------------------------------------------------------------------------
# 1-2. Test metrics + moneyness buckets
# ---------------------------------------------------------------------------
test_df = pd.read_csv(f"{DATA_DIR}/test_split.csv")
y_true = test_df[TARGET].values
y_pred = predict(test_df)
err = y_pred - y_true
abs_err = np.abs(err)

mae = abs_err.mean()
rmse = np.sqrt((err ** 2).mean())
max_err = abs_err.max()
max_err_row = test_df.iloc[abs_err.argmax()]

print("\n--- Test set metrics ---")
print(f"MAE:      {mae:.4f}")
print(f"RMSE:     {rmse:.4f}")
print(f"Max err:  {max_err:.4f}  (at S0={max_err_row['S0']:.1f}, K={max_err_row['K']:.1f}, "
      f"T={max_err_row['T']:.2f}, r={max_err_row['r']:.3f}, sigma={max_err_row['sigma']:.2f}, "
      f"true price={max_err_row['price']:.2f})")

moneyness = test_df["S0"] / test_df["K"]  # put ITM when S0 < K, i.e. moneyness < 1
buckets = pd.cut(moneyness, bins=[0, 0.85, 1.15, np.inf],
                  labels=["deep ITM (S0/K<0.85)", "near ATM (0.85-1.15)", "deep OTM (S0/K>1.15)"])
bucket_df = pd.DataFrame({"bucket": buckets, "abs_err": abs_err})
bucket_mae = bucket_df.groupby("bucket", observed=True)["abs_err"].agg(["mean", "count"])
bucket_mae.columns = ["MAE", "n_samples"]
print("\n--- MAE by moneyness bucket ---")
print(bucket_mae)

metrics = {
    "test_MAE": float(mae),
    "test_RMSE": float(rmse),
    "test_max_abs_error": float(max_err),
    "bucket_MAE": bucket_mae["MAE"].to_dict(),
    "bucket_n": bucket_mae["n_samples"].to_dict(),
}
with open(f"{CKPT_DIR}/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------------------
# 3. Predicted vs binomial scatter
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, s=6, alpha=0.35, color="#2563eb")
lims = [0, max(y_true.max(), y_pred.max()) * 1.02]
plt.plot(lims, lims, "r--", linewidth=1.2, label="y = x (perfect match)")
plt.xlabel("Binomial (CRR) price — ground truth")
plt.ylabel("NN predicted price")
plt.title("Predicted vs Binomial American Put Price (test set)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/scatter_pred_vs_binomial.png", dpi=150)
plt.close()
print(f"\nSaved scatter plot -> {PLOT_DIR}/scatter_pred_vs_binomial.png")

# ---------------------------------------------------------------------------
# 4. NN surface vs binomial surface: fixed K=100, r=5%, sigma=25%
# ---------------------------------------------------------------------------
K_fix, r_fix, sigma_fix = 100.0, 0.05, 0.25
S0_grid = np.linspace(50, 150, 60)
T_grid = np.linspace(0.05, 2.0, 60)
S0_mesh, T_mesh = np.meshgrid(S0_grid, T_grid)

binom_surface = np.zeros_like(S0_mesh)
for i in range(S0_mesh.shape[0]):
    for j in range(S0_mesh.shape[1]):
        binom_surface[i, j] = crr_put_price(S0_mesh[i, j], K_fix, T_mesh[i, j],
                                             r_fix, sigma_fix, steps=200, american=True)

nn_surface = predict_raw(
    S0_mesh.ravel(), np.full(S0_mesh.size, K_fix), T_mesh.ravel(),
    np.full(S0_mesh.size, r_fix), np.full(S0_mesh.size, sigma_fix)
).reshape(S0_mesh.shape)

diff_surface = nn_surface - binom_surface

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, surf, title, cmap in zip(
        axes, [binom_surface, nn_surface, diff_surface],
        ["Binomial surface", "NN surface", "NN - Binomial (error)"],
        ["viridis", "viridis", "RdBu_r"]):
    if cmap == "RdBu_r":
        vmax = np.abs(surf).max()
        im = ax.pcolormesh(S0_grid, T_grid, surf, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    else:
        im = ax.pcolormesh(S0_grid, T_grid, surf, cmap=cmap, shading="auto")
    ax.set_xlabel("S0")
    ax.set_ylabel("T (years)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
fig.suptitle(f"Put price surface: K={K_fix:.0f}, r={r_fix:.0%}, sigma={sigma_fix:.0%}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/surface_comparison.png", dpi=150)
plt.close()
print(f"Saved surface comparison -> {PLOT_DIR}/surface_comparison.png")

surface_max_abs_err = float(np.abs(diff_surface).max())
surface_mean_abs_err = float(np.abs(diff_surface).mean())
print(f"Surface comparison: mean abs err {surface_mean_abs_err:.4f}, "
      f"max abs err {surface_max_abs_err:.4f}")

# ---------------------------------------------------------------------------
# 5. Finance sanity checks
# ---------------------------------------------------------------------------
print("\n--- Finance sanity checks ---")

# (a) non-negativity of NN predictions across the whole test set + surface grid
neg_test = (y_pred < -1e-6).sum()
neg_surface = (nn_surface < -1e-6).sum()
print(f"(a) Non-negativity: {neg_test} negative predictions on test set, "
      f"{neg_surface} negative points on surface grid.")

# (b) monotonic decreasing in S0 (put price should fall as spot rises, K/T/r/sigma fixed)
S0_line = np.linspace(50, 150, 200)
mono_price = predict_raw(S0_line, np.full(200, K_fix), np.full(200, 1.0),
                          np.full(200, r_fix), np.full(200, sigma_fix))
diffs = np.diff(mono_price)
n_violations = int((diffs > 1e-3).sum())  # small numerical tolerance
print(f"(b) Monotonicity in S0 (K=100,T=1,r=5%,sigma=25%): "
      f"{n_violations}/{len(diffs)} increasing steps (violations) out of {len(diffs)} steps.")

plt.figure(figsize=(6, 4))
plt.plot(S0_line, mono_price, color="#16a34a")
plt.xlabel("S0")
plt.ylabel("NN predicted put price")
plt.title("Monotonicity check: put price vs S0 (K=100, T=1, r=5%, sigma=25%)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/monotonicity_check.png", dpi=150)
plt.close()

# (c) intrinsic value floor violations on test set (allow small numerical slack)
intrinsic_test = np.maximum(test_df["K"].values - test_df["S0"].values, 0.0)
floor_violation = y_pred < (intrinsic_test - 0.25)  # 0.25 slack for NN noise
n_floor_violation = int(floor_violation.sum())
worst_floor_violation = float((intrinsic_test - y_pred)[floor_violation].max()) if n_floor_violation else 0.0
print(f"(c) Intrinsic value floor: {n_floor_violation}/{len(test_df)} predictions more than "
      f"$0.25 below intrinsic value" + (f" (worst breach: ${worst_floor_violation:.2f})" if n_floor_violation else "") + ".")

sanity_results = {
    "negative_predictions_test": int(neg_test),
    "negative_predictions_surface_grid": int(neg_surface),
    "monotonicity_violations": n_violations,
    "monotonicity_steps_checked": len(diffs),
    "intrinsic_floor_violations": n_floor_violation,
    "intrinsic_floor_worst_breach": worst_floor_violation,
    "surface_mean_abs_error": surface_mean_abs_err,
    "surface_max_abs_error": surface_max_abs_err,
}
with open(f"{CKPT_DIR}/sanity_results.json", "w") as f:
    json.dump(sanity_results, f, indent=2)

print("\nSaved metrics -> checkpoints/test_metrics.json")
print("Saved sanity results -> checkpoints/sanity_results.json")
