"""Standardised operator interface (proposal Eq. 20):

    operator.execute(I, x_t, params) -> (C_t, c_t, metadata)

Every operator returns a list of candidate tours, the compute it actually used
(in move evaluations) and a metadata dict. It must never use more than
``max_evals``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OpResult:
    candidates: list[np.ndarray]
    cost: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Operator:
    name: str = "base"
    family: str = "base"        # e.g. destroy_repair, kick, local, restart, multi
    intensity: float = 0.0      # q_t in Eq. (7)
    branching: int = 1          # m_t in Eq. (7)

    def execute(self, inst, x: np.ndarray, rng: np.random.Generator, max_evals: int) -> OpResult:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}(family={self.family}, q={self.intensity}, m={self.branching})"
