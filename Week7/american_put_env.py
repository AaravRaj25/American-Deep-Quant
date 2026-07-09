"""
american_put_env.py

American put option exercise problem formulated as a Markov Decision Process (MDP)
and implemented as a small Gym-style environment.

MDP definition (see report for full explanation)
--------------------------------------------------
State (observation), s_t:
    [moneyness, time_fraction_remaining, intrinsic_value_scaled]
    - moneyness              = S_t / K
    - time_fraction_remaining= (n_steps - t) / n_steps        in [0, 1]
    - intrinsic_value_scaled = max(K - S_t, 0) / K
    All three are computable from information available at time t only
    (current price, current step index, and the fixed strike). No future
    prices, no post-decision information, and no realized-path summary
    statistics beyond the current price are included, so the state cannot
    leak information about what will happen after t.

Actions:
    0 = HOLD      -> keep the option alive, no payoff, price evolves one step
    1 = EXERCISE  -> receive intrinsic payoff max(K - S_t, 0) immediately, episode ends

Transitions:
    - After HOLD (and t is not the last step): price moves one step forward
      under risk-neutral GBM, t increments by 1, reward = 0, episode continues.
    - After HOLD at the last step (t = n_steps - 1): the option is forced to
      terminate at expiry. This mirrors an American option that was never
      exercised -> its payoff at expiry equals a European payoff. reward =
      max(K - S_T, 0), episode ends.
    - After EXERCISE (any t): reward = max(K - S_t, 0) using the CURRENT
      price (not a future one), episode ends immediately. The option cannot
      be exercised twice because done=True blocks further step() calls.

Reward timing & discounting:
    - The environment pays reward exactly once, at the terminal step
      (either exercise or forced expiry). All intermediate HOLD steps pay 0.
      This matches the "reward paid once" requirement and avoids double
      counting.
    - The environment itself does NOT discount the reward internally -- it
      returns the raw (undiscounted) cash payoff at t, plus info['step']
      telling you which step this happened at. Discounting to time 0 is a
      property of how you compute returns/objectives, so it is applied
      by the caller (see discounted_payoff() below) using
      gamma = exp(-r * dt) per step. This separation keeps the environment
      re-usable for both discounted and undiscounted analyses.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


HOLD = 0
EXERCISE = 1


@dataclass
class AmericanPutEnv:
    S0: float = 100.0        # initial underlying price
    K: float = 100.0         # strike price
    r: float = 0.03          # risk-free rate (annualized, continuously compounded)
    sigma: float = 0.2       # volatility (annualized)
    T: float = 1.0           # time to maturity (years)
    n_steps: int = 50        # number of discrete decision points
    seed: int | None = None

    def __post_init__(self):
        self.dt = self.T / self.n_steps
        self._rng = np.random.default_rng(self.seed)
        self.t = 0
        self.S = self.S0
        self.done = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new episode. Returns the initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.S = self.S0
        self.done = False
        return self._get_state()

    def step(self, action: int):
        """
        Advance the environment by one action.

        Returns: (next_state, reward, done, info)
        info contains: {'reason': 'exercise'|'expiry'|None, 'step': int, 'price': float}
        Raises RuntimeError if called after the episode has already ended,
        which is exactly the "cannot step after done" invariant.
        """
        if self.done:
            raise RuntimeError(
                "step() called after episode termination. Call reset() to start a new episode."
            )
        if action not in (HOLD, EXERCISE):
            raise ValueError(f"Invalid action {action!r}; expected 0 (hold) or 1 (exercise).")

        if action == EXERCISE:
            payoff = self._intrinsic(self.S)
            self.done = True
            info = {"reason": "exercise", "step": self.t, "price": self.S}
            return self._get_state(), payoff, True, info

        # action == HOLD
        if self.t >= self.n_steps - 1:
            # Forced termination at expiry: American option held to maturity
            # behaves like a European option at this final instant.
            payoff = self._intrinsic(self.S)
            self.done = True
            info = {"reason": "expiry", "step": self.t, "price": self.S}
            return self._get_state(), payoff, True, info

        # Ordinary hold: advance price by one step of risk-neutral GBM.
        z = self._rng.standard_normal()
        self.S = self.S * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * z
        )
        self.t += 1
        info = {"reason": None, "step": self.t, "price": self.S}
        return self._get_state(), 0.0, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _intrinsic(self, S: float) -> float:
        return max(self.K - S, 0.0)

    def _get_state(self) -> np.ndarray:
        moneyness = self.S / self.K
        time_frac = (self.n_steps - self.t) / self.n_steps
        intrinsic_scaled = self._intrinsic(self.S) / self.K
        return np.array([moneyness, time_frac, intrinsic_scaled], dtype=np.float32)

    def discount_factor(self, step: int) -> float:
        """Discount factor exp(-r * step * dt) from time 0 to a given step index."""
        return float(np.exp(-self.r * step * self.dt))


def discounted_payoff(env: AmericanPutEnv, raw_payoff: float, terminal_step: int) -> float:
    """Apply time-0 discounting to a raw terminal payoff paid at terminal_step."""
    return raw_payoff * env.discount_factor(terminal_step)
