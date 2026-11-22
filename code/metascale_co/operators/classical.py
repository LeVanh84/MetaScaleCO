"""Classical CPU operators standing in for neural search operators.

Mapping to the proposal's GenSCO operator library (Sec. 5.2, Eq. 14-19):
    light    ~ o(1) light disruption + one generation cycle   (destroy-repair, 10 %)
    strong   ~ o(2) strong disruption + one generation cycle  (destroy-repair, 30 %)
    multi    ~ o(3) multi-candidate generation                (4 x light, keep best)
    oropt    ~ o(4) cheap local improvement                   (or-opt on incumbent)
    kick     ~ o(5) restart from diversified incumbent        (double-bridge + 2-opt)
    restart  ~ full restart                                   (random insertion + 2-opt)
    STOP     ~ o(6)
"""
from __future__ import annotations

import numpy as np

from ..problems import tsp
from .base import Operator, OpResult


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


class DestroyRepair(Operator):
    family = "destroy_repair"

    def __init__(self, name: str, ratio: float):
        self.name = name
        self.intensity = ratio

    def execute(self, inst, x, rng, max_evals):
        cand, ev, ok = tsp.destroy_repair(x, inst.D, inst.neigh, self.intensity, _seed(rng), max_evals)
        return OpResult([cand] if ok else [], int(ev), {"complete": bool(ok)})


class MultiSample(Operator):
    family = "multi"

    def __init__(self, name: str, ratio: float, m: int):
        self.name = name
        self.intensity = ratio
        self.branching = m

    def execute(self, inst, x, rng, max_evals):
        cands, used = [], 0
        for _ in range(self.branching):
            cand, ev, ok = tsp.destroy_repair(x, inst.D, inst.neigh, self.intensity, _seed(rng),
                                              max_evals - used)
            used += int(ev)
            if ok:
                cands.append(cand)
            if used >= max_evals:
                break
        return OpResult(cands, used, {"n_candidates": len(cands)})


class OrOpt(Operator):
    family = "local"

    def __init__(self, name: str = "oropt"):
        self.name = name

    def execute(self, inst, x, rng, max_evals):
        cand, ev = tsp.or_opt(x, inst.D, inst.neigh, max_evals)
        return OpResult([cand], int(ev))


class Kick(Operator):
    family = "kick"

    def __init__(self, name: str = "kick", n_kicks: int = 1):
        self.name = name
        self.intensity = n_kicks

    def execute(self, inst, x, rng, max_evals):
        y = tsp.double_bridge(x, _seed(rng), int(self.intensity))
        cand, ev = tsp.two_opt(y, inst.D, inst.neigh, max_evals)
        return OpResult([cand], int(ev))


class Restart(Operator):
    family = "restart"

    def __init__(self, name: str = "restart"):
        self.name = name
        self.intensity = 1.0

    def execute(self, inst, x, rng, max_evals):
        y, ev = tsp.random_insertion_tour(inst.D, _seed(rng))
        if ev >= max_evals:
            return OpResult([], int(max_evals), {"complete": False})
        cand, e2 = tsp.two_opt(y, inst.D, inst.neigh, max_evals - ev)
        return OpResult([cand], int(ev + e2))


def default_library() -> list[Operator]:
    """Operator library (order defines action ids)."""
    return [
        DestroyRepair("light", 0.10),
        DestroyRepair("strong", 0.30),
        MultiSample("multi", 0.10, 4),
        OrOpt("oropt"),
        Kick("kick", 1),
        Restart("restart"),
    ]


OP_NAMES = [op.name for op in default_library()]
