"""Instance sets, LKH reference solutions and the compute unit C0.

Usage:  uv run python -m metascale_co.data.datasets
"""
from __future__ import annotations

from multiprocessing import Pool

import elkai
import numpy as np

from ..config import ART, load_config, write_json, read_json
from ..problems import tsp
from ..environments.budgeted_search_env import initial_solution
from ..operators import get_library


def make_coords(n, dist, count, seed):
    rng = np.random.default_rng(seed)
    f = tsp.uniform_instance if dist == "uniform" else tsp.clustered_instance
    return np.stack([f(n, rng) for _ in range(count)])


def _lkh(args):
    coords, runs = args
    D = tsp.distance_matrix(coords)
    # elkai works on integer-like precision internally; scale coordinates
    c = elkai.Coordinates2D({i: (float(x) * 1e4, float(y) * 1e4) for i, (x, y) in enumerate(coords)})
    tour = np.array(c.solve_tsp(runs=runs)[:-1], dtype=np.int64)
    assert tsp.is_valid_tour(tour, len(coords))
    return tsp.tour_length(tour, D)


def build_sets():
    cfg = load_config()
    ms = cfg["master_seed"]
    sets = {}
    tr = cfg["train"]
    sets["train"] = [make_coords(n, "uniform", tr["n_per_size"], ms * 10 + n) for n in tr["sizes"]]
    v = cfg["val"]
    sets["val"] = make_coords(v["n"], "uniform", v["count"], ms * 10 + 7)
    for name, t in cfg["test"].items():
        sets[name] = make_coords(t["n"], t["dist"], t["count"], ms * 1000 + t["seed_offset"])
    for i, arr in enumerate(sets["train"]):
        np.save(ART / "data" / f"train_{i}.npy", arr)
    names = ["val"] + list(cfg["test"].keys())
    with Pool(8) as pool:
        for name in names:
            arr = sets[name]
            ref = np.array(pool.map(_lkh, [(c, cfg["lkh_runs"]) for c in arr]))
            np.savez(ART / "data" / f"{name}.npz", coords=arr, ref=ref)
            print(f"{name}: {arr.shape}, mean LKH length {ref.mean():.4f}")


def calibrate_c0():
    """C0(n) = mean cost of one 'light' operator applied to x0 (one standard operation)."""
    cfg = load_config()
    sizes = sorted({*cfg["train"]["sizes"], cfg["val"]["n"], *[t["n"] for t in cfg["test"].values()]})
    light = get_library()[0]
    out = {}
    for n in sizes:
        arr = make_coords(n, "uniform", 20, 99991 + n)
        costs = []
        for j, c in enumerate(arr):
            inst = tsp.TSPInstance(c)
            x0 = initial_solution(inst, j)
            rng = np.random.default_rng(j)
            light.execute(inst, x0, rng, 10**12)          # warm-up (JIT compile)
            costs += [light.execute(inst, x0, rng, 10**12).cost for _ in range(10)]
        out[str(n)] = float(np.mean(costs))
        print(f"C0(n={n}) = {out[str(n)]:.1f} move evaluations")
    write_json("c0.json", out)


def load_set(name):
    z = np.load(ART / "data" / f"{name}.npz")
    return z["coords"], z["ref"]


def c0_of(n: int) -> float:
    return read_json("c0.json")[str(n)]


if __name__ == "__main__":
    calibrate_c0()
    build_sets()
