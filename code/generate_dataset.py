"""
generate_dataset.py
Week 6, Part A: synthetic American put dataset generation.

Samples contracts over documented, realistic ranges, labels them with the
Week 4 CRR pricer at a FIXED tree step count, runs label sanity checks,
and saves the dataset to CSV so training can be rerun without
regenerating labels.
"""
import time
import numpy as np
import pandas as pd
from crr_pricer import crr_put_price

SEED = 42
N_SAMPLES = 12000
TREE_STEPS = 200  # fixed step count for ALL labels -> consistent supervision signal

# ---------------------------------------------------------------------------
# Feature ranges (documented)
# ---------------------------------------------------------------------------
# S0    : spot price            U[50, 150]      -- generic equity price scale
# K     : strike price          U[50, 150]      -- same scale as S0 so S0/K
#                                                   moneyness spans ~0.33-3.0
# T     : time to maturity      U[0.05, 2.0]    -- 2 weeks to 2 years
# r     : risk-free rate        U[0.00, 0.10]   -- 0% to 10% annual, continuous
# sigma : volatility            U[0.05, 0.60]   -- 5% (low-vol blue chip) to
#                                                   60% (high-vol growth/small cap)
RANGES = {
    "S0": (50.0, 150.0),
    "K": (50.0, 150.0),
    "T": (0.05, 2.0),
    "r": (0.00, 0.10),
    "sigma": (0.05, 0.60),
}


def sample_contracts(n, seed=SEED):
    rng = np.random.default_rng(seed)
    S0 = rng.uniform(*RANGES["S0"], size=n)
    K = rng.uniform(*RANGES["K"], size=n)
    T = rng.uniform(*RANGES["T"], size=n)
    r = rng.uniform(*RANGES["r"], size=n)
    sigma = rng.uniform(*RANGES["sigma"], size=n)
    return S0, K, T, r, sigma


def label_dataset(S0, K, T, r, sigma, steps=TREE_STEPS):
    n = len(S0)
    prices = np.empty(n, dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        prices[i] = crr_put_price(S0[i], K[i], T[i], r[i], sigma[i],
                                   steps=steps, american=True)
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  labeled {i+1}/{n} contracts ({elapsed:.1f}s elapsed)")
    return prices


def sanity_check(df):
    print("\n--- Label sanity checks ---")
    n_total = len(df)

    finite_mask = np.isfinite(df["price"])
    n_finite = finite_mask.sum()
    print(f"Finite prices: {n_finite}/{n_total}")
    assert n_finite == n_total, "Non-finite prices found!"

    nonneg_mask = df["price"] >= -1e-8
    n_nonneg = nonneg_mask.sum()
    print(f"Non-negative prices: {n_nonneg}/{n_total}")
    assert n_nonneg == n_total, "Negative prices found!"

    intrinsic = np.maximum(df["K"] - df["S0"], 0.0)
    violation = df["price"] < (intrinsic - 1e-6)
    n_violation = violation.sum()
    print(f"Rows below intrinsic value: {n_violation}/{n_total}")
    assert n_violation == 0, "American put priced below intrinsic value!"

    print("All label sanity checks PASSED.\n")


def main():
    print(f"Generating {N_SAMPLES} synthetic American put contracts "
          f"(fixed tree steps = {TREE_STEPS})...")
    S0, K, T, r, sigma = sample_contracts(N_SAMPLES)

    print("Labeling with CRR binomial pricer (this takes a little while)...")
    prices = label_dataset(S0, K, T, r, sigma, steps=TREE_STEPS)

    df = pd.DataFrame({
        "S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "price": prices
    })

    sanity_check(df)

    out_path = "/home/claude/week6/data/american_put_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")
    print(df.describe())


if __name__ == "__main__":
    main()
