"""
run_experiment.py

Entry point for Week 7 deliverables:
  - Part B.5: run 5 sample episodes and print exercise/expiry reasons.
  - Part C.1-C.3: compare always-hold, immediate-exercise, and random
    policies over >=1000 episodes; report average (discounted) payoff and
    exercise rate; save a histogram of exercise timing.
  - Part C.4 (optional): train the tabular Q-learning sketch and plot its
    learned exercise region.

Usage:
    python3 run_experiment.py

Reproducibility:
    A single top-level SEED controls every random draw in this script
    (episode price paths and any policy randomness). Change SEED below to
    reproduce with a different stream, or pass --seed on the command line.
"""

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from american_put_env import AmericanPutEnv, discounted_payoff, HOLD, EXERCISE
from policies import (
    always_hold_policy,
    immediate_exercise_policy,
    random_policy,
    train_tabular_q,
)

SEED = 2026
ENV_PARAMS = dict(S0=100.0, K=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=50)
N_EVAL_EPISODES = 2000


def run_episode(env: AmericanPutEnv, policy_fn, rng, episode_seed):
    state = env.reset(seed=episode_seed)
    done = False
    info = None
    while not done:
        action = policy_fn(state, env, rng)
        state, raw_payoff, done, info = env.step(action)
    d_payoff = discounted_payoff(env, raw_payoff, info["step"])
    return {
        "raw_payoff": raw_payoff,
        "discounted_payoff": d_payoff,
        "reason": info["reason"],
        "step": info["step"],
        "exercised": info["reason"] == "exercise",
    }


def print_sample_episodes(n=5):
    print(f"\n=== {n} sample episodes (policy = immediate_exercise) ===")
    env = AmericanPutEnv(**ENV_PARAMS, seed=SEED)
    rng = np.random.default_rng(SEED)
    for i in range(n):
        result = run_episode(env, immediate_exercise_policy, rng, episode_seed=SEED + i)
        print(
            f"episode {i}: reason={result['reason']:8s} step={result['step']:3d} "
            f"raw_payoff={result['raw_payoff']:.4f} discounted_payoff={result['discounted_payoff']:.4f}"
        )


def evaluate_policy(name, policy_fn, n_episodes=N_EVAL_EPISODES, seed=SEED):
    env = AmericanPutEnv(**ENV_PARAMS, seed=seed)
    rng = np.random.default_rng(seed)
    raw_payoffs, disc_payoffs, exercised_flags, exercise_steps = [], [], [], []
    for i in range(n_episodes):
        result = run_episode(env, policy_fn, rng, episode_seed=seed + 10_000 + i)
        raw_payoffs.append(result["raw_payoff"])
        disc_payoffs.append(result["discounted_payoff"])
        exercised_flags.append(result["exercised"])
        if result["exercised"]:
            exercise_steps.append(result["step"])
    return {
        "name": name,
        "n_episodes": n_episodes,
        "avg_raw_payoff": float(np.mean(raw_payoffs)),
        "avg_discounted_payoff": float(np.mean(disc_payoffs)),
        "exercise_rate": float(np.mean(exercised_flags)),
        "exercise_steps": exercise_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--train-q", action="store_true", help="also train the optional tabular Q sketch")
    args = parser.parse_args()

    seed = args.seed

    print(f"Environment params: {ENV_PARAMS}")
    print(f"Seed: {seed}, evaluation episodes per policy: {args.episodes}")

    # Part B.5
    print_sample_episodes(n=5)

    # Part C.1-C.3
    policies = [
        ("always_hold", always_hold_policy),
        ("immediate_exercise", immediate_exercise_policy),
        ("random_p05", lambda s, e, r: random_policy(s, e, r, p_exercise=0.05)),
    ]

    results = [evaluate_policy(name, fn, n_episodes=args.episodes, seed=seed) for name, fn in policies]

    print(f"\n=== Policy comparison over {args.episodes} episodes ===")
    print(f"{'policy':20s} {'avg_raw_payoff':>15s} {'avg_disc_payoff':>16s} {'exercise_rate':>14s}")
    for r in results:
        print(f"{r['name']:20s} {r['avg_raw_payoff']:15.4f} {r['avg_discounted_payoff']:16.4f} {r['exercise_rate']:14.3f}")

    # Save table to CSV
    with open("policy_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "n_episodes", "avg_raw_payoff", "avg_discounted_payoff", "exercise_rate"])
        for r in results:
            writer.writerow([r["name"], r["n_episodes"], r["avg_raw_payoff"], r["avg_discounted_payoff"], r["exercise_rate"]])
    print("\nSaved table -> policy_comparison.csv")

    # Plot: exercise-time histograms for the nontrivial policies (Part C.3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, r in zip(axes, [res for res in results if res["name"] != "always_hold"]):
        steps = r["exercise_steps"]
        ax.hist(steps, bins=min(30, ENV_PARAMS["n_steps"]), color="#4C72B0", edgecolor="white")
        ax.set_title(f"{r['name']}\nexercise_rate={r['exercise_rate']:.2f}")
        ax.set_xlabel("step index at exercise")
        ax.set_ylabel("count")
        ax.set_xlim(0, ENV_PARAMS["n_steps"])
    fig.suptitle("When exercise happens (nontrivial policies)")
    fig.tight_layout()
    fig.savefig("exercise_timing.png", dpi=150)
    print("Saved plot -> exercise_timing.png")

    # Optional Part C.4: tabular Q-learning sketch
    if args.train_q:
        print("\n=== Training optional tabular Q-learning sketch ===")
        train_env = AmericanPutEnv(**ENV_PARAMS, seed=seed)
        q_policy = train_tabular_q(train_env, n_episodes=20000, seed=seed)
        q_result = evaluate_policy("tabular_q", q_policy, n_episodes=args.episodes, seed=seed)
        print(
            f"{'tabular_q':20s} {q_result['avg_raw_payoff']:15.4f} "
            f"{q_result['avg_discounted_payoff']:16.4f} {q_result['exercise_rate']:14.3f}"
        )

        # Visualize the learned exercise region: for each (moneyness, time)
        # bin, greedy action of the trained policy.
        money_grid = np.linspace(0.5, 1.5, q_policy.n_money_bins)
        time_grid = np.linspace(0, 1, q_policy.n_time_bins)
        region = np.zeros((q_policy.n_money_bins, q_policy.n_time_bins))
        for i, m in enumerate(money_grid):
            for j, t in enumerate(time_grid):
                region[i, j] = np.argmax(q_policy.Q[i, j])

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(
            region, origin="lower", aspect="auto",
            extent=[time_grid.min(), time_grid.max(), money_grid.min(), money_grid.max()],
            cmap="RdBu_r", vmin=0, vmax=1,
        )
        ax2.set_xlabel("time fraction remaining")
        ax2.set_ylabel("moneyness (S/K)")
        ax2.set_title("Learned exercise region (tabular Q)\nred=exercise, blue=hold")
        fig2.colorbar(im, ax=ax2, ticks=[0, 1], label="0=hold, 1=exercise")
        fig2.tight_layout()
        fig2.savefig("learned_exercise_region.png", dpi=150)
        print("Saved plot -> learned_exercise_region.png")

        with open("policy_comparison.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                q_result["name"], q_result["n_episodes"], q_result["avg_raw_payoff"],
                q_result["avg_discounted_payoff"], q_result["exercise_rate"],
            ])


if __name__ == "__main__":
    main()
