"""Meta-level policies: MetaScale-CO controller (Sec. 5.5-5.6) and baselines (Sec. 9.2)."""
from __future__ import annotations

import numpy as np

from ..environments.budgeted_search_env import STOP
from .state_encoder import encode_state


class Policy:
    name = "policy"
    deployable = True

    def reset(self, env):
        pass

    def act(self, env) -> int:
        raise NotImplementedError


class FixedPolicy(Policy):
    """Fixed scaling: repeat one operator until the budget is exhausted."""

    def __init__(self, op_id: int, name: str):
        self.op_id, self.name = op_id, name

    def act(self, env):
        return self.op_id


class RandomPolicy(Policy):
    name = "Random"

    def reset(self, env):
        self.rng = np.random.default_rng(env.rng.integers(1 << 30))

    def act(self, env):
        return int(self.rng.integers(len(env.ops)))


class StagnationPolicy(Policy):
    """Heuristic: light refinement; after k non-improving steps apply a kick."""
    name = "Stagnation"

    def __init__(self, light: int, kick: int, k: int = 3):
        self.light, self.kick, self.k = light, kick, k

    def act(self, env):
        return self.kick if env.h.steps_since_improve >= self.k else self.light


class UCB1AOS(Policy):
    """Classic adaptive operator selection (online bandit, no offline training).
    Reward = gain (%) per C0 unit of compute, normalised by the running max."""
    name = "UCB1-AOS"

    def __init__(self, c: float = 0.5):
        self.c = c

    def reset(self, env):
        A = len(env.ops)
        self.n = np.zeros(A)
        self.s = np.zeros(A)
        self.rmax = 1e-9
        self._last = None

    def act(self, env):
        h = env.h
        if self._last is not None and h.actions and h.actions[-1] == self._last:
            r = 100.0 * h.gains[-1] / max(h.costs[-1] / env.C0, 1e-3)
            self.rmax = max(self.rmax, r)
            self.n[self._last] += 1
            self.s[self._last] += r
        untried = np.where(self.n == 0)[0]
        if len(untried):
            a = int(untried[0])
        else:
            mean = self.s / self.n / self.rmax
            a = int(np.argmax(mean + self.c * np.sqrt(2 * np.log(self.n.sum()) / self.n)))
        self._last = a
        return a


class MetaScalePolicy(Policy):
    """MetaScale-CO controller.

    U_beta(s,a) = mu(s,a) + beta * sigma(s,a) - lambda * c~(s,a)      (Eq. 27)
    a_t = argmax over feasible actions                               (Eq. 28)
    STOP if max_a U <= tau_stop                                      (Eq. 29)
    """

    def __init__(self, voc, name: str, beta: float = 0.0, lam: float = 0.0,
                 sigma: str = "ale", history: bool = True, allowed=None,
                 tau_stop: float | None = None, log_decisions: bool = False):
        self.voc, self.name = voc, name
        self.beta, self.lam, self.sigma = beta, lam, sigma
        self.history, self.allowed = history, allowed
        self.tau_stop = tau_stop
        self.log_decisions = log_decisions
        self.decisions = []

    def reset(self, env):
        self.decisions = []

    def scores(self, env):
        A = len(env.ops)
        acts = np.arange(A) if self.allowed is None else np.asarray(self.allowed)
        z = encode_state(env, history=self.history)
        X = np.concatenate([np.repeat(z[None], len(acts), 0), np.eye(A, dtype=np.float32)[acts]], 1)
        mu, s_ale, s_epi, cost = self.voc.predict(X)
        sig = s_ale if self.sigma == "ale" else s_epi
        U = mu + self.beta * sig - self.lam * cost
        return acts, U, mu, sig, cost

    def act(self, env):
        acts, U, mu, sig, cost = self.scores(env)
        feasible = cost * env.C0 <= max(env.b, 0)
        Uf = np.where(feasible, U, -np.inf) if feasible.any() else U
        k = int(np.argmax(Uf))
        a = int(acts[k])
        if self.tau_stop is not None and U[k] <= self.tau_stop:
            a = STOP
        if self.log_decisions:
            self.decisions.append((env.t, 1.0 - env.b / env.B, a, float(mu[k]), float(sig[k]),
                                   float(cost[k])))
        return a


class OracleOneStep(Policy):
    """Upper bound (not deployable): tries every operator from the same state and
    random stream, keeps the one with the best realised gain per unit cost."""
    name = "Oracle-1step"
    deployable = False
