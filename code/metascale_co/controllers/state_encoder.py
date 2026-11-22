"""Meta-state encoder z_t = [phi_I(I), phi_x(x_t), phi_h(h_t), b_t / B]  (Eq. 6).

All features are dimensionless (Eq. 22-23) so that a controller trained on
TSP-50/100 can be applied to TSP-200 and clustered instances.
"""
from __future__ import annotations

import numpy as np

N_STATIC = 5      # instance + budget descriptors (used by the difficulty-only ablation)


def encode_state(env, history: bool = True) -> np.ndarray:
    inst, h = env.inst, env.h
    n_ops = len(env.ops)
    B_units = env.B / env.C0
    b_units = max(env.b, 0) / env.C0
    static = [
        1.0 - env.b / env.B,                       # budget fraction used
        np.log2(1.0 + B_units) / 8.0,              # total budget scale
        np.log2(1.0 + b_units) / 8.0,              # remaining budget scale
        inst.n / 100.0,                            # size
        inst.cv_nn,                                # geometry / clustering descriptor
    ]
    if not history:
        dyn = np.zeros(4 + 3 * n_ops)
        return np.concatenate([static, dyn]).astype(np.float32)
    g = np.array(h.gains[-3:]) * 100.0 if h.gains else np.zeros(1)
    dyn = [
        (h.gains[-1] * 100.0) if h.gains else 0.0,     # delta_t   (Eq. 22, in %)
        float(g.mean()),                               # mean delta over K=3 (Eq. 23)
        min(h.steps_since_improve, 30) / 10.0,         # stagnation length
        100.0 * (env.f0 - env.f) / env.f0,             # total progress since x0 (%)
    ]
    tried = (h.op_count > 0).astype(float)
    usage = h.op_count / max(1, h.op_count.sum())
    return np.concatenate([static, dyn, np.clip(h.op_ema_rate, -1, 20) * tried,
                           usage, np.clip(h.op_last_gain, -1, 20)]).astype(np.float32)


def state_dim(n_ops: int) -> int:
    return N_STATIC + 4 + 3 * n_ops


def action_onehot(a: int, n_ops: int) -> np.ndarray:
    v = np.zeros(n_ops, np.float32)
    v[a] = 1.0
    return v
