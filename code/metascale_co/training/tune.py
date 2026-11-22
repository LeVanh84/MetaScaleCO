"""Tune every method with the same protocol on the validation set (TSP-100 uniform,
64 instances, budgets {4, 16, 64} C0): pick the configuration with the lowest
mean gap. Fixed-tuned gets a separate best operator for *each* budget, which is
a stronger baseline than any single global setting.

Usage:  uv run python -m metascale_co.training.tune
"""
from __future__ import annotations

import itertools
import numpy as np

from ..config import load_config, write_json
from ..controllers.policies import FixedPolicy, UCB1AOS
from ..data.datasets import c0_of, load_set
from ..environments.budgeted_search_env import initial_solution
from ..evaluation.runner import run_episode
from ..methods import ms_policy
from ..operators import get_library
from ..operators.classical import OP_NAMES
from ..parallel import pmap
from ..problems import tsp

_CTX = {}


def _init():
    coords, ref = load_set("val")
    _CTX["insts"] = [tsp.TSPInstance(c) for c in coords]
    _CTX["ref"] = ref
    _CTX["x0"] = [initial_solution(I, 50_000 + j) for j, I in enumerate(_CTX["insts"])]


def _eval(args):
    kind, params, budgets = args
    ops = get_library()
    gaps = []
    for j, I in enumerate(_CTX["insts"]):
        C0 = c0_of(I.n)
        for B in budgets:
            if kind == "fixed":
                pol = FixedPolicy(OP_NAMES.index(params), params)
            elif kind == "ucb1":
                pol = UCB1AOS(params)
            else:
                pol = ms_policy(kind, params[0], params[1])
            r = run_episode(I, ops, pol, B, C0, 50_000 + j, _CTX["x0"][j])
            gaps.append(100 * (r["f"] - _CTX["ref"][j]) / _CTX["ref"][j])
    return float(np.mean(gaps))


def main():
    cfg = load_config()
    g = cfg["grid"]
    tb = cfg["val"]["tune_budgets"]
    tuned = {"fixed": {}, "ms": {}, "grid_scores": {}}
    W = cfg["workers"]

    def run(tasks):
        return pmap(_eval, tasks, W, _init)

    # Fixed-tuned: best operator for every evaluation budget
    for B in cfg["budgets"]:
        scores = run([("fixed", op, [B]) for op in OP_NAMES])
        tuned["fixed"][str(B)] = OP_NAMES[int(np.argmin(scores))]
        print(f"Fixed-tuned B={B:3d}: {dict(zip(OP_NAMES, np.round(scores, 3)))}")
    scores = run([("ucb1", c, tb) for c in g["ucb1_c"]])
    tuned["ucb1_c"] = g["ucb1_c"][int(np.argmin(scores))]
    print(f"UCB1-AOS c: {dict(zip(g['ucb1_c'], np.round(scores, 3)))} -> {tuned['ucb1_c']}")
    for variant in ["MS-Myopic", "MS-UCB", "MS-UCB-epi", "MS-DiffOnly", "MS-2op"]:
        if variant == "MS-Myopic":
            betas = [0.0]                                   # no uncertainty term
        elif variant in ("MS-UCB", "MS-UCB-epi"):
            betas = [b for b in g["beta"] if b > 0]         # UCB variants: beta > 0
        else:
            betas = list(g["beta"])
        grid = list(itertools.product(g["lambda"], betas))
        scores = run([(variant, p, tb) for p in grid])
        k = int(np.argmin(scores))
        tuned["ms"][variant] = {"lam": grid[k][0], "beta": grid[k][1], "val_gap": scores[k]}
        tuned["grid_scores"][variant] = [[*p, s] for p, s in zip(grid, scores)]
        print(f"{variant:12s} best lam={grid[k][0]} beta={grid[k][1]} val gap={scores[k]:.4f}%")
    # the proposed variant used as reference in the statistics = best on validation
    proposed = ["MS-Myopic", "MS-UCB", "MS-UCB-epi"]
    tuned["reference"] = min(proposed, key=lambda v: tuned["ms"][v]["val_gap"])
    print(f"reference method (best proposed variant on validation): {tuned['reference']}")
    write_json("tuned.json", tuned)


if __name__ == "__main__":
    main()
