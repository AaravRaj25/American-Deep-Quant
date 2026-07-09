# Week 7 — American Put RL Environment

## Files
- `american_put_env.py` — the `AmericanPutEnv` class (MDP + environment). Full MDP definition (state, actions, transitions, reward timing, discounting, no-leakage argument) is documented in its module docstring.
- `policies.py` — `always_hold_policy`, `immediate_exercise_policy`, `random_policy`, plus an optional `TabularQPolicy` / `train_tabular_q` sketch for Part C.4.
- `test_american_put_env.py` — 5 invariant tests (non-negative payoff, cannot step after done, terminates at expiry, reward paid exactly once, state has no future info).
- `run_experiment.py` — runs 5 sample episodes, evaluates all policies over 2000 episodes, saves `policy_comparison.csv` and plots.

## Run it
```bash
python3 test_american_put_env.py            # invariant tests
python3 run_experiment.py                    # sample episodes + 3-policy comparison
python3 run_experiment.py --train-q          # also trains the optional tabular Q sketch
python3 run_experiment.py --seed 123 --episodes 5000   # custom seed / episode count
```

## Parameters used
`S0=100, K=100, r=0.03, sigma=0.20, T=1.0 year, n_steps=50` (weekly-ish decision points), seed `2026`.

## Results (2000 episodes/policy, seed=2026)

| policy | avg raw payoff | avg discounted payoff | exercise rate |
|---|---|---|---|
| always_hold | 6.4704 | 6.2829 | 0.000 |
| immediate_exercise | 1.8225 | 1.8186 | 0.904 |
| random (p=0.05/step) | 3.6627 | 3.6090 | 0.918 |
| tabular_q (optional, 20k training episodes) | 4.1398 | 4.1175 | 0.766 |

See `exercise_timing.png` for when the nontrivial policies exercise, and `learned_exercise_region.png` for the tabular-Q sketch's learned hold/exercise boundary over (moneyness, time-to-maturity).

## Reflection

At-the-money (S0=K=100) with meaningful volatility (σ=0.2) and a full year to expiry, **always-hold-to-expiry** gets the highest average payoff among the three fixed baselines. That is expected here: the put currently has zero intrinsic value, so exercising now (or at the first random opportunity) locks in either nothing or a small, likely-suboptimal payoff, while holding preserves the option's full time value and lets the stock wander further into the money before a decision is forced at expiry. **Immediate-exercise** is the worst policy — it exercises whenever the option has *any* positive intrinsic value, even a few cents, which throws away all remaining time value and matches none of the standard American-put intuition from Week 4 (early exercise is only optimal when the put is *deep* in the money and time value is small, e.g., close to expiry or with a high discount rate making immediate cash preferable). **Random** does better than immediate-exercise simply by exercising less eagerly on average, but it still exercises without reference to moneyness or time, so it's not a reasoned policy.

The optional **tabular Q** policy lands between always-hold and the naive policies, and its learned region (see the heatmap) qualitatively shows exercise concentrated at high moneyness (deep ITM, S well below K) and late in the episode (little time remaining) — consistent with Week 4's early-exercise boundary intuition, though this coarse, short-trained sketch is not yet competitive with always-hold in this at-the-money, long-maturity setting. That gap — a smarter policy should beat always-hold when the put is *not* purely dominated by time value — is exactly the target for Week 8's stronger trained agent.

## No-leakage note
The state `[moneyness, time_fraction_remaining, intrinsic_value_scaled]` is recomputed at every step purely from the environment's *current* price `S_t` and step index `t`. The environment does not pre-reveal any portion of the future price path to the agent; each new price is sampled only at the moment `step()` is called for a HOLD action. `test_state_has_no_future_information` in the test file checks this by reconstructing the returned state from `(S_t, t)` alone.
