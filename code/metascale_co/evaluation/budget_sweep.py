"""Main benchmark: every method x every budget x 3 test sets (ID, size shift,
geometry shift) x 100 instances x 3 seeds. Also records
  * anytime traces at the largest budget (Sec. 9.5),
  * VoC calibration pairs (predicted mu vs realised gain, Fig. 4 of the proposal),
  * learned-stopping runs with a sweep over tau_stop (Eq. 29).

Usage:  uv run python -m metascale_co.evaluation.budget_sweep
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from ..config import RES, load_config, read_json
from ..data.datasets import c0_of, load_set
from ..environments.budgeted_search_env import initial_solution
from .. import methods as M
from ..operators import get_library
from ..operators.classical import OP_NAMES
from ..parallel import pmap
from ..problems import tsp
from .runner import run_episode

GRID = 129     # anytime grid: 0..128 C0 units

_CTX = {}


def _init(dataset, stop_only=False):
    _CTX["stop_only"] = stop_only
    coords, ref = load_set(dataset)
    _CTX.update(dataset=dataset, insts=[tsp.TSPInstance(c) for c in coords], ref=ref,
                cfg=load_config(), tuned=read_json("tuned.json"))


def _anytime(trace, C0, ref, Bmax):
    """Gap of the incumbent on an integer grid of compute (C0 units)."""
    used = np.array([u for u, _ in trace]) / C0
    f = np.array([v for _, v in trace])
    grid = np.arange(GRID) * (Bmax / (GRID - 1))
    idx = np.searchsorted(used, grid, side="right") - 1
    return (100 * (f[idx] - ref) / ref).astype(np.float32)


def _task(args):
    j, s = args
    cfg, tuned, ds = _CTX["cfg"], _CTX["tuned"], _CTX["dataset"]
    I, ref = _CTX["insts"][j], _CTX["ref"][j]
    C0 = c0_of(I.n)
    ops = get_library()
    seed = 1_000_000 * (1 + list(cfg["test"]).index(ds)) + 1000 * j + s
    x0 = initial_solution(I, seed)
    f0 = tsp.tour_length(x0, I.D)
    Bmax = max(cfg["budgets"])
    rows, traces, calib = [], {}, []

    def record(name, B, r, extra=None):
        acts = np.bincount(np.array(r["actions"], dtype=int), minlength=len(OP_NAMES))
        share = acts / max(1, acts.sum())
        row = dict(dataset=ds, budget=B, algorithm=name, instance=j, seed=s, f=r["f"], ref=ref,
                   gap=100 * (r["f"] - ref) / ref, gap0=100 * (f0 - ref) / ref,
                   used_units=r["used_units"], steps=r["steps"], wall_ms=1000 * r["wall"])
        row.update({f"share_{o}": v for o, v in zip(OP_NAMES, share)})
        if extra:
            row.update(extra)
        rows.append(row)

    for name in ([] if _CTX.get("stop_only") else M.ALL):
        for B in cfg["budgets"]:
            big = B == Bmax
            pol = M.build(name, tuned, B, log=big and name.startswith("MS-"))
            r = run_episode(I, ops, pol, B, C0, seed, x0, record_trace=big)
            record(name, B, r)
            if big:
                traces[name] = _anytime(r["trace"], C0, ref, Bmax)
                for mu, g, sg, a, frac in r["calib"]:
                    calib.append((ds, name, j, s, mu, g, sg, OP_NAMES[a], frac))
    # learned stopping: tau sweep on the validation-selected proposed variant
    ref_name = tuned["reference"]
    p = tuned["ms"][ref_name]
    for tau in cfg["stop_taus"]:
        pol = M.ms_policy(ref_name, p["lam"], p["beta"], tau=tau, name=f"{ref_name}+STOP")
        r = run_episode(I, ops, pol, Bmax, C0, seed, x0)
        record(f"{ref_name}+STOP", Bmax, r, {"tau": tau})
    return rows, traces, calib


def main(stop_only=False):
    """stop_only=True re-runs only the learned-stopping tau sweep and replaces those rows."""
    cfg = load_config()
    tuned = read_json("tuned.json")
    print(f"reference (validation-selected) method: {tuned['reference']}")
    all_rows, all_calib, any_traces = [], [], {}
    for ds, t in cfg["test"].items():
        t0 = time.time()
        tasks = [(j, s) for j in range(t["count"]) for s in range(cfg["seeds_per_instance"])]
        out = pmap(_task, tasks, cfg["workers"], _init, (ds, stop_only), chunksize=2)
        for rows, traces, calib in out:
            all_rows += rows; all_calib += calib
            for k, v in traces.items():
                any_traces.setdefault(f"{ds}|{k}", []).append(v)
        print(f"{ds}: {len(tasks)} (instance, seed) tasks in {time.time() - t0:.0f}s")
    df = pd.DataFrame(all_rows)
    if stop_only:
        old = pd.read_csv(RES / "results.csv")
        old = old[~old.algorithm.str.endswith("+STOP")]
        pd.concat([old, df], ignore_index=True).to_csv(RES / "results.csv", index=False)
        print(f"replaced stopping rows: {len(df)}")
        return
    df.to_csv(RES / "results.csv", index=False)
    pd.DataFrame(all_calib, columns=["dataset", "algorithm", "instance", "seed", "mu", "gain",
                                     "sigma", "op", "budget_frac"]).to_csv(RES / "calibration.csv",
                                                                          index=False)
    np.savez_compressed(RES / "anytime.npz", **{k: np.stack(v) for k, v in any_traces.items()})
    print(f"wrote {len(df)} result rows")


if __name__ == "__main__":
    import sys
    main(stop_only="--stop-only" in sys.argv)
