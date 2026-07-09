"""
test_american_put_env.py

Minimal invariant tests for AmericanPutEnv, runnable with:
    python3 test_american_put_env.py
(no pytest dependency required, but pytest will also pick these up
 if it's installed, since functions are named test_*).
"""

import numpy as np
from american_put_env import AmericanPutEnv, HOLD, EXERCISE


def test_payoff_is_never_negative():
    """Invariant: exercise/expiry payoff must always be >= 0."""
    env = AmericanPutEnv(S0=100, K=100, r=0.03, sigma=0.3, T=1.0, n_steps=20, seed=0)
    rng = np.random.default_rng(1)
    for episode in range(200):
        env.reset(seed=episode)
        done = False
        while not done:
            # bias towards exercising early/randomly to hit many price levels,
            # including deep out-of-the-money and deep in-the-money cases
            action = EXERCISE if rng.random() < 0.3 else HOLD
            state, reward, done, info = env.step(action)
            assert reward >= 0.0, f"Negative payoff {reward} at episode {episode}, step {info['step']}"
    print("PASS: test_payoff_is_never_negative")


def test_cannot_step_after_done():
    """Invariant: calling step() after the episode has terminated must raise."""
    env = AmericanPutEnv(S0=100, K=100, r=0.03, sigma=0.2, T=1.0, n_steps=10, seed=42)
    env.reset()
    env.step(EXERCISE)  # terminates the episode immediately
    assert env.done is True
    try:
        env.step(HOLD)
    except RuntimeError:
        pass
    else:
        raise AssertionError("step() did not raise RuntimeError after episode was done")
    print("PASS: test_cannot_step_after_done")


def test_episode_terminates_on_expiry_if_never_exercised():
    """Invariant: always-hold must terminate exactly at the final step (expiry)."""
    env = AmericanPutEnv(S0=100, K=100, r=0.03, sigma=0.2, T=1.0, n_steps=15, seed=7)
    env.reset()
    done = False
    steps_taken = 0
    info = None
    while not done:
        _, _, done, info = env.step(HOLD)
        steps_taken += 1
        assert steps_taken <= env.n_steps, "Episode ran longer than n_steps without terminating"
    assert info["reason"] == "expiry"
    assert info["step"] == env.n_steps - 1
    print("PASS: test_episode_terminates_on_expiry_if_never_exercised")


def test_reward_paid_exactly_once():
    """Invariant: only the terminal transition pays a nonzero reward."""
    env = AmericanPutEnv(S0=100, K=100, r=0.03, sigma=0.2, T=1.0, n_steps=15, seed=3)
    env.reset()
    done = False
    nonzero_rewards = 0
    while not done:
        _, reward, done, _ = env.step(HOLD)
        if reward != 0.0:
            nonzero_rewards += 1
    assert nonzero_rewards == 1, f"Expected exactly 1 nonzero reward, got {nonzero_rewards}"
    print("PASS: test_reward_paid_exactly_once")


def test_state_has_no_future_information():
    """
    Invariant: the state returned at time t must be reproducible from only
    (S_t, t, K, n_steps) -- i.e. it must not depend on any price sampled
    after t. We check this by re-deriving the state from the env's own
    current price/step and comparing to what was returned.
    """
    env = AmericanPutEnv(S0=100, K=100, r=0.03, sigma=0.25, T=1.0, n_steps=12, seed=5)
    state = env.reset()
    done = False
    while not done:
        expected = np.array([
            env.S / env.K,
            (env.n_steps - env.t) / env.n_steps,
            max(env.K - env.S, 0.0) / env.K,
        ], dtype=np.float32)
        assert np.allclose(state, expected), "Returned state does not match current (S_t, t) only"
        state, reward, done, info = env.step(HOLD)
    print("PASS: test_state_has_no_future_information")


if __name__ == "__main__":
    test_payoff_is_never_negative()
    test_cannot_step_after_done()
    test_episode_terminates_on_expiry_if_never_exercised()
    test_reward_paid_exactly_once()
    test_state_has_no_future_information()
    print("\nAll tests passed.")
