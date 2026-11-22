"""Batch-level global-budget allocation experiment (Sec. 5.7).

Compares, for batches of N=10 test instances sharing N x B C0 units:
  * Uniform split  : every instance gets B units, solved independently by the
                     reference MetaScale policy (and by Fixed-tuned);
  * MS-Batch       : one controller allocates the pooled budget across instances.

Usage:  uv run python -m metascale_co.evaluation.batch_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RES, load_config, read_json
from ..data.datasets import c0_of, load_set
from ..environments.batch_budget_env import BatchBudgetEnv, run_batch
from ..environments.budgeted_search_env import initial_solution
from .. import methods as M
from ..operators import get_library
from ..parallel import pmap
from ..problems import tsp
from .runner import run_episode

_CTX = {}


def _init(ds):
    coords, ref = load_set(ds)
    _CTX.update(ds=ds, insts=[tsp.TSPInstance(c) for c in coords], ref=ref,
                cfg=load_config(), tuned=read_json("tuned.json"))


def _task(args):
    b, s, B = args
    cfg, tuned, ds = _CTX["cfg"], _CTX["tuned"], _CTX["ds"]
    N = cfg["batch"]["size"]
    idx = list(range(b * N, (b + 1) * N))
    insts = [_CTX["insts"][j] for j in idx]
    refs = _CTX["ref"][idx]
    C0s = [c0_of(I.n) for I in insts]
    seeds = [7_000_000 + 1000 * j + s for j in idx]
    x0s = [initial_solution(I, sd) for I, sd in zip(insts, seeds)]
    f0 = np.array([tsp.tour_length(x, I.D) for x, I in zip(x0s, insts)])
    ref_name = tuned["reference"]
    p = tuned["ms"][ref_name]
    rows = []
    ops = get_library()
    for name in ["Fixed-tuned", ref_name]:
        fs = []
        for I, c0, sd, x in zip(insts, C0s, seeds, x0s):
            pol = M.build(name, tuned, B)
            fs.append(run_episode(I, ops, pol, B, c0, sd, x)["f"])
        for k, j in enumerate(idx):
            rows.append(dict(dataset=ds, budget=B, method=f"{name} (uniform split)", batch=b, seed=s,
                             instance=j, gap=100 * (fs[k] - refs[k]) / refs[k],
                             gap0=100 * (f0[k] - refs[k]) / refs[k], spent=B))
    benv = BatchBudgetEnv(insts, ops, B, C0s, seeds, x0s)
    pol = M.ms_policy(ref_name, p["lam"], p["beta"])
    fs, spent = run_batch(benv, pol)
    for k, j in enumerate(idx):
        rows.append(dict(dataset=ds, budget=B, method="MS-Batch (pooled)", batch=b, seed=s,
                         instance=j, gap=100 * (fs[k] - refs[k]) / refs[k],
                         gap0=100 * (f0[k] - refs[k]) / refs[k], spent=spent[k]))
    return rows


def main():
    cfg = load_config()
    rows = []
    for ds in cfg["test"]:
        nb = cfg["test"][ds]["count"] // cfg["batch"]["size"]
        tasks = [(b, s, B) for b in range(nb) for s in range(cfg["seeds_per_instance"])
                 for B in cfg["batch"]["budgets"]]
        for r in pmap(_task, tasks, cfg["workers"], _init, (ds,)):
            rows += r
        print(f"{ds}: done")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "batch_results.csv", index=False)
    print(df.groupby(["dataset", "budget", "method"]).gap.mean().round(3).unstack())


if __name__ == "__main__":
    main()
