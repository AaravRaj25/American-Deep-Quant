"""
crr_pricer.py
Week 4 CRR binomial tree pricer, reused as the labeling function for Week 6.

Vectorized backward induction over the terminal stock price vector.
"""
import numpy as np


def crr_put_price(S0, K, T, r, sigma, steps=200, american=True):
    """
    Price a European or American put option using a Cox-Ross-Rubinstein
    binomial tree with vectorized backward induction.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Spot, strike, time to maturity (years), risk-free rate (annual,
        continuously compounded), and annualized volatility.
    steps : int
        Number of tree steps (fixed across the dataset for label consistency).
    american : bool
        If True, allow early exercise at every node (American put).
        If False, price a European put (early exercise check skipped).

    Returns
    -------
    float : option price at t=0
    """
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        # degenerate case: deterministic forward price
        ST = S0 * np.exp(r * T)
        return max(K - ST, 0.0) * np.exp(-r * T)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    # guard against numerical edge cases pushing p outside [0,1]
    p = min(max(p, 0.0), 1.0)

    # terminal stock prices: S0 * u^j * d^(N-j) for j = 0..N
    j = np.arange(steps + 1)
    ST = S0 * (u ** j) * (d ** (steps - j))
    values = np.maximum(K - ST, 0.0)

    # backward induction
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])
        if american:
            j = np.arange(i + 1)
            S_i = S0 * (u ** j) * (d ** (i - j))
            intrinsic = np.maximum(K - S_i, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


def crr_put_price_batch(S0, K, T, r, sigma, steps=200, american=True):
    """Convenience wrapper to price many contracts (loop over vectorized calls)."""
    out = np.empty(len(S0), dtype=np.float64)
    for idx in range(len(S0)):
        out[idx] = crr_put_price(S0[idx], K[idx], T[idx], r[idx], sigma[idx],
                                  steps=steps, american=american)
    return out


if __name__ == "__main__":
    # quick smoke test against a known-ish sanity case
    price = crr_put_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=200, american=True)
    print(f"ATM American put (S=K=100, T=1, r=5%, sigma=25%): {price:.4f}")
