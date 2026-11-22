"""Budgeted meta-level search environment (proposal Sec. 4, Algorithm 1).

State s_t = (I, x_t, h_t, b_t); action = operator id or STOP; the incumbent is
updated with Eq. (9) and the budget with b_{t+1} = b_t - c_t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..problems import tsp

STOP = -1


@dataclass
class History:
    n_ops: int
    gains: list[float] = field(default_factory=list)          # normalised gain G_t per step
    actions: list[int] = field(default_factory=list)
    costs: list[int] = field(default_factory=list)
    steps_since_improve: int = 0
    op_count: np.ndarray = None
    op_ema_rate: np.ndarray = None   # EMA of gain-per-unit-cost for each operator
    op_last_gain: np.ndarray = None

    def __post_init__(self):
        self.op_count = np.zeros(self.n_ops)
        self.op_ema_rate = np.zeros(self.n_ops)
        self.op_last_gain = np.full(self.n_ops, -1.0)   # -1 = never tried


class BudgetedSearchEnv:
    def __init__(self, inst: tsp.TSPInstance, ops, budget: float, C0: float,
                 seed: int, x0: np.ndarray | None = None, record_trace: bool = False):
        self.inst, self.ops, self.C0 = inst, ops, float(C0)
        self.B = int(round(budget))
        self.rng = np.random.default_rng(seed)
        if x0 is None:
            x0 = initial_solution(inst, seed)
        self.x0 = x0
        self.f0 = tsp.tour_length(x0, inst.D)
        self.x = x0.copy()
        self.f = self.f0
        self.b = self.B
        self.t = 0
        self.h = History(len(ops))
        self.done = False
        self.record_trace = record_trace
        self.trace = [(0, self.f)] if record_trace else None   # (compute used, incumbent length)

    # ----------------------------------------------------------------- core step
    def step(self, a: int):
        """Execute action a. Returns (gain G_t, cost c_t)."""
        if a == STOP or self.b <= 0:
            self.done = True
            return 0.0, 0
        res = self.ops[a].execute(self.inst, self.x, self.rng, self.b)
        cost = int(res.cost)
        f_old = self.f
        for c in res.candidates:                       # Eq. (9)
            fc = tsp.tour_length(c, self.inst.D)
            if fc < self.f - 1e-12:
                self.f, self.x = fc, c
        gain = (f_old - self.f) / (abs(f_old) + 1e-12)  # Eq. (24)
        self._update_history(a, gain, cost)
        self.b -= cost
        self.t += 1
        if self.b <= 0 or cost == 0:      # cost==0 can only happen with no budget left
            self.done = True
        if self.record_trace:
            self.trace.append((self.B - self.b, self.f))
        return gain, cost

    def _update_history(self, a, gain, cost):
        h = self.h
        h.gains.append(gain)
        h.actions.append(a)
        h.costs.append(cost)
        h.steps_since_improve = 0 if gain > 0 else h.steps_since_improve + 1
        h.op_count[a] += 1
        rate = 100.0 * gain / max(cost / self.C0, 1e-3)
        h.op_ema_rate[a] = rate if h.op_count[a] == 1 else 0.7 * h.op_ema_rate[a] + 0.3 * rate
        h.op_last_gain[a] = 100.0 * gain

    def snapshot(self):
        """Copy of the mutable search state (used by the one-step oracle)."""
        import copy
        return copy.deepcopy((self.x, self.f, self.b, self.t, self.h, self.rng.bit_generator.state))

    def restore(self, snap):
        import copy
        x, f, b, t, h, rs = copy.deepcopy(snap)
        self.x, self.f, self.b, self.t, self.h = x, f, b, t, h
        self.rng.bit_generator.state = rs

    @property
    def used(self) -> int:
        return self.B - self.b


def initial_solution(inst: tsp.TSPInstance, seed: int) -> np.ndarray:
    """Base solver initialisation x0: nearest neighbour from a seeded start + 2-opt.
    Not charged to the budget (identical for every method)."""
    start = int(np.random.default_rng(seed + 7919).integers(inst.n))
    x = tsp.nearest_neighbor_tour(inst.D, start)
    x, _ = tsp.two_opt(x, inst.D, inst.neigh, 10**12)
    return x
