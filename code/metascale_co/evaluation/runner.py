"""Algorithm 1 (proposal p. 9): run one policy on one instance under budget B."""
from __future__ import annotations

import time

import numpy as np

from ..environments.budgeted_search_env import STOP, BudgetedSearchEnv
from ..controllers.policies import OracleOneStep


def run_episode(inst, ops, policy, B_units: float, C0: float, seed: int, x0,
                record_trace: bool = False):
    env = BudgetedSearchEnv(inst, ops, B_units * C0, C0, seed, x0=x0, record_trace=record_trace)
    policy.reset(env)
    calib = []          # (predicted mu %, realised gain %, sigma) for MetaScale policies
    t0 = time.perf_counter()
    while not env.done and env.b > 0:
        if isinstance(policy, OracleOneStep):
            a = _oracle_choice(env)
        else:
            a = policy.act(env)
        if a == STOP:
            break
        g, c = env.step(a)
        if getattr(policy, "log_decisions", False) and policy.decisions:
            d = policy.decisions[-1]
            calib.append((d[3], 100.0 * g, d[4], a, d[1]))
        if c == 0:
            break
    wall = time.perf_counter() - t0
    return {
        "f": env.f, "f0": env.f0, "used": env.used, "used_units": env.used / C0,
        "steps": env.t, "wall": wall, "actions": list(env.h.actions),
        "trace": env.trace, "calib": calib,
    }


def _oracle_choice(env):
    snap = env.snapshot()
    n_trace = len(env.trace) if env.trace is not None else 0
    best_a, best_score, best_cost = 0, -np.inf, np.inf
    for a in range(len(env.ops)):
        g, c = env.step(a)
        score = g / max(c, 1)
        if score > best_score or (score == best_score and c < best_cost):
            best_a, best_score, best_cost = a, score, c
        env.restore(snap)
        env.done = False
        if env.trace is not None:
            del env.trace[n_trace:]
    return best_a       # executing it after restore reproduces the realised outcome
