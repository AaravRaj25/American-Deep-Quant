# Week 6 — Neural Network Pricer for American Puts

MLP regressor trained to match CRR binomial American put prices (Week 4 pricer).

## Structure
```
code/
  crr_pricer.py          # Week 4 CRR binomial pricer (vectorized backward induction)
  generate_dataset.py    # Part A: samples + labels 12,000 contracts, sanity checks
  train_mlp.py            # Part B: 80/10/10 split, train-only standardization, MLP training
  evaluate.py              # Part C: metrics, moneyness buckets, scatter, surface, sanity checks
  plot_learning_curve.py   # train/val MSE vs epoch plot
  build_report.py          # assembles Week6_Aarav.pdf
data/
  american_put_dataset.csv # full labeled dataset (generated)
  test_split.csv            # held-out test split (generated, for reproducible eval)
checkpoints/
  best_mlp.pt               # best-validation-MSE model weights
  scaler.json                # train-set feature mean/std
  history.json                # per-epoch train/val MSE
  test_metrics.json            # MAE/RMSE/max err + bucket MAE
  sanity_results.json           # finance sanity check results
plots/
  learning_curve.png
  scatter_pred_vs_binomial.png
  surface_comparison.png
  monotonicity_check.png
```

## Setup
```bash
pip install torch numpy pandas scikit-learn matplotlib reportlab
```

## Run (clean session, in order)
```bash
cd code
python generate_dataset.py       # ~40s, writes data/american_put_dataset.csv
python train_mlp.py              # trains MLP, writes checkpoints/
python evaluate.py               # writes plots/ + metrics
python plot_learning_curve.py    # writes plots/learning_curve.png
python build_report.py           # writes Week6_Aarav.pdf
```

## Reproducibility
Single seed (42) controls: contract sampling, train/val/test split, PyTorch
weight init, and DataLoader shuffling.

## Key results (test set, n=1200)
- MAE: 0.0883, RMSE: 0.1230, Max abs error: 1.0538
- MAE by moneyness: deep ITM 0.097, near ATM 0.114, deep OTM 0.061
- Monotonicity in S0: 0/199 violations
- Non-negativity: 119/1200 test predictions slightly negative (near-zero-price
  deep OTM contracts — output layer has no floor at 0)
- Intrinsic value floor: 8/1200 predictions breach by >$0.25 (worst $1.05)

See Week6_Aarav.pdf for full analysis and reflection.
