"""Batch-level allocation (proposal Sec. 5.7, Eq. 31): N instances share one global
budget B_tot; at each decision the controller picks (instance, action) with the
largest U. Each instance keeps its own search state; the budget features of every
instance are set from the *global* remaining budget per instance.
"""
from __future__ import annotations

import numpy as np

from .budgeted_search_env import BudgetedSearchEnv


class BatchBudgetEnv:
    def __init__(self, insts, ops, per_instance_units: float, C0s, seeds, x0s):
        self.N = len(insts)
        self.C0s = np.asarray(C0s, float)
        # budget accounting in C0 units so that instances of different sizes are comparable
        self.total_units = per_instance_units * self.N
        self.left_units = self.total_units
        self.envs = [BudgetedSearchEnv(I, ops, per_instance_units * c0, c0, s, x0=x)
                     for I, c0, s, x in zip(insts, self.C0s, seeds, x0s)]
        self.spent_units = np.zeros(self.N)

    def _sync_budget_features(self):
        per = max(self.left_units, 0.0) / self.N
        for e, c0 in zip(self.envs, self.C0s):
            e.b = int(per * c0)

    def step(self, i: int, a: int):
        e = self.envs[i]
        self._sync_budget_features()
        e.b = int(max(self.left_units, 0.0) * self.C0s[i])   # may spend up to the global pool
        g, c = e.step(a)
        e.done = False
        u = c / self.C0s[i]
        self.left_units -= u
        self.spent_units[i] += u
        self._sync_budget_features()
        return g, c

    @property
    def done(self):
        return self.left_units <= 1e-9


def run_batch(benv: BatchBudgetEnv, policy):
    """Global argmax of U over (instance, action) pairs."""
    for e in benv.envs:
        policy.reset(e)
    benv._sync_budget_features()
    while not benv.done:
        best = (-np.inf, 0, 0)
        for i, e in enumerate(benv.envs):
            acts, U, mu, sig, cost = policy.scores(e)
            feas = cost <= benv.left_units
            if not feas.any():
                continue
            k = int(np.argmax(np.where(feas, U, -np.inf)))
            if U[k] > best[0]:
                best = (U[k], i, int(acts[k]))
        if best[0] == -np.inf:   # nothing fits in the remaining pool
            break
        _, i, a = best
        g, c = benv.step(i, a)
        if c == 0:
            break
    return np.array([e.f for e in benv.envs]), benv.spent_units.copy()
