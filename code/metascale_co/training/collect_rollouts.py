"""Stage 1 (Sec. 6.2): build the trajectory bank with diverse behaviour schedules:
uniform random operator choice, every fixed operator, and the stagnation heuristic.

Usage:  uv run python -m metascale_co.training.collect_rollouts
"""
from __future__ import annotations

import numpy as np

from ..config import ART, load_config
from ..controllers.policies import FixedPolicy, RandomPolicy, StagnationPolicy
from ..data.datasets import c0_of
from ..data.trajectory_logger import TrajectoryLogger
from ..environments.budgeted_search_env import BudgetedSearchEnv, initial_solution
from ..operators import get_library
from ..operators.classical import OP_NAMES
from ..parallel import pmap
from ..problems import tsp


def _behaviours():
    A = len(OP_NAMES)
    pols = [RandomPolicy(), RandomPolicy()]
    pols += [FixedPolicy(a, f"Fixed-{OP_NAMES[a]}") for a in range(A)]
    pols += [StagnationPolicy(OP_NAMES.index("light"), OP_NAMES.index("kick"))]
    return pols


def _work(args):
    coords, inst_id, budgets = args
    inst = tsp.TSPInstance(coords)
    C0 = c0_of(inst.n)
    ops = get_library()
    log = TrajectoryLogger()
    rng = np.random.default_rng(inst_id)
    for p_id, pol in enumerate(_behaviours()):
        seed = int(rng.integers(1 << 30))
        B = float(rng.choice(budgets))           # budget curriculum, Eq. (37)
        env = BudgetedSearchEnv(inst, ops, B * C0, C0, seed, x0=initial_solution(inst, seed))
        pol.reset(env)
        while not env.done and env.b > 0:
            a = pol.act(env)
            log.before(env)
            g, c = env.step(a)
            log.after(env, a, g, c, inst_id * 100 + p_id)
            if c == 0:
                break
    return log.arrays()


def main():
    cfg = load_config()
    tasks, k = [], 0
    for i in range(len(cfg["train"]["sizes"])):
        for c in np.load(ART / "data" / f"train_{i}.npy"):
            tasks.append((c, k, cfg["train"]["behavior_budgets"])); k += 1
    parts = pmap(_work, tasks, cfg["workers"], chunksize=4)
    bank = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    np.savez(ART / "trajectory_bank.npz", **bank)
    print(f"trajectory bank: {len(bank['a'])} transitions from {len(tasks)} instances")
    for a, name in enumerate(OP_NAMES):
        m = bank["a"] == a
        print(f"  {name:8s} n={m.sum():6d}  mean gain {bank['g'][m].mean():.3f}%  "
              f"P(gain>0)={np.mean(bank['g'][m] > 0):.2f}  mean cost {bank['c'][m].mean():.2f} C0")


if __name__ == "__main__":
    main()
