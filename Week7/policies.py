"""
policies.py

Three baseline policies over AmericanPutEnv, plus a tiny tabular
Q-learning sketch (optional / Part C.4).

Every policy is a callable: policy(state, env, rng) -> action in {0, 1}
so they can all be run through the same run_episode() harness.
"""

import numpy as np
from american_put_env import AmericanPutEnv, HOLD, EXERCISE


def always_hold_policy(state, env, rng):
    """Never exercise voluntarily; payoff only realized (if any) at expiry."""
    return HOLD


def immediate_exercise_policy(state, env, rng):
    """Exercise as soon as the option is in the money (intrinsic > 0); else hold."""
    moneyness, time_frac, intrinsic_scaled = state
    return EXERCISE if intrinsic_scaled > 0 else HOLD


def random_policy(state, env, rng, p_exercise: float = 0.05):
    """Exercise with small fixed probability at each decision point, else hold."""
    return EXERCISE if rng.random() < p_exercise else HOLD


# ----------------------------------------------------------------------
# Optional: tabular Q-learning sketch (Part C.4)
# ----------------------------------------------------------------------

class TabularQPolicy:
    """
    Discretizes (moneyness, time_fraction) into bins and learns a
    HOLD-vs-EXERCISE Q-table with vanilla tabular Q-learning.

    This is intentionally simple: it is a *sketch* to visualize a learned
    exercise region, not a production-quality solver. Week 8 will replace
    this with a stronger trained agent.
    """

    def __init__(self, n_money_bins=20, n_time_bins=20,
                 money_range=(0.5, 1.5), alpha=0.1, gamma_step=None, epsilon=0.1):
        self.n_money_bins = n_money_bins
        self.n_time_bins = n_time_bins
        self.money_range = money_range
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma_step = gamma_step  # set from env.discount factor per step
        # Q[money_bin, time_bin, action]
        self.Q = np.zeros((n_money_bins, n_time_bins, 2))

    def _bin(self, moneyness, time_frac):
        lo, hi = self.money_range
        m_idx = int(np.clip((moneyness - lo) / (hi - lo) * self.n_money_bins, 0, self.n_money_bins - 1))
        t_idx = int(np.clip(time_frac * self.n_time_bins, 0, self.n_time_bins - 1))
        return m_idx, t_idx

    def act(self, state, rng, greedy=False):
        moneyness, time_frac, _ = state
        m_idx, t_idx = self._bin(moneyness, time_frac)
        if (not greedy) and rng.random() < self.epsilon:
            return rng.integers(0, 2)
        return int(np.argmax(self.Q[m_idx, t_idx]))

    def __call__(self, state, env, rng):
        return self.act(state, rng, greedy=True)

    def update(self, state, action, reward, next_state, done):
        moneyness, time_frac, _ = state
        m_idx, t_idx = self._bin(moneyness, time_frac)
        if done:
            target = reward
        else:
            nm, nt, _ = next_state
            nm_idx, nt_idx = self._bin(nm, nt)
            target = reward + self.gamma_step * np.max(self.Q[nm_idx, nt_idx])
        td_error = target - self.Q[m_idx, t_idx, action]
        self.Q[m_idx, t_idx, action] += self.alpha * td_error


def train_tabular_q(env: AmericanPutEnv, n_episodes=20000, seed=0):
    rng = np.random.default_rng(seed)
    policy = TabularQPolicy(gamma_step=np.exp(-env.r * env.dt))
    for ep in range(n_episodes):
        state = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        while not done:
            action = policy.act(state, rng, greedy=False)
            next_state, reward, done, info = env.step(action)
            policy.update(state, action, reward, next_state, done)
            state = next_state
    return policy
