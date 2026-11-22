"""Builds every compared method from tuned hyper-parameters (artifacts/tuned.json)."""
from __future__ import annotations

from .config import ART
from .controllers.policies import (FixedPolicy, MetaScalePolicy, OracleOneStep, RandomPolicy,
                                   StagnationPolicy, UCB1AOS)
from .controllers.voc_model import NumpyVoC
from .operators.classical import OP_NAMES

LIGHT, STRONG, KICK = OP_NAMES.index("light"), OP_NAMES.index("strong"), OP_NAMES.index("kick")

# display order + groups (used by tables and plots)
BASELINES = ["Fixed-light", "Fixed-tuned", "Random", "Stagnation", "UCB1-AOS"]
PROPOSED = ["MS-Myopic", "MS-UCB", "MS-UCB-epi"]
ABLATIONS = ["MS-DiffOnly", "MS-2op"]
BOUND = ["Oracle-1step"]
ALL = BASELINES + PROPOSED + ABLATIONS + BOUND

_VOC = {}


def voc(tag):
    if tag not in _VOC:
        _VOC[tag] = NumpyVoC(ART / "models" / f"voc_{tag}.npz")
    return _VOC[tag]


def ms_policy(variant: str, lam: float, beta: float, tau=None, log=False, name=None):
    kw = dict(lam=lam, beta=beta, tau_stop=tau, log_decisions=log, name=name or variant)
    if variant == "MS-Myopic":
        return MetaScalePolicy(voc("full"), beta=0.0, lam=lam, tau_stop=tau,
                               log_decisions=log, name=name or variant)
    if variant == "MS-UCB":
        return MetaScalePolicy(voc("full"), sigma="ale", **kw)
    if variant == "MS-UCB-epi":
        return MetaScalePolicy(voc("full"), sigma="epi", **kw)
    if variant == "MS-DiffOnly":
        return MetaScalePolicy(voc("static"), sigma="ale", history=False, **kw)
    if variant == "MS-2op":
        return MetaScalePolicy(voc("full"), sigma="ale", allowed=[LIGHT, STRONG], **kw)
    raise KeyError(variant)


def build(name: str, tuned: dict, budget: int, log: bool = False):
    if name == "Fixed-light":
        return FixedPolicy(LIGHT, name)
    if name == "Fixed-tuned":
        return FixedPolicy(OP_NAMES.index(tuned["fixed"][str(budget)]), name)
    if name == "Random":
        return RandomPolicy()
    if name == "Stagnation":
        return StagnationPolicy(LIGHT, KICK)
    if name == "UCB1-AOS":
        return UCB1AOS(tuned["ucb1_c"])
    if name == "Oracle-1step":
        return OracleOneStep()
    p = tuned["ms"][name]
    return ms_policy(name, p["lam"], p["beta"], log=log)
