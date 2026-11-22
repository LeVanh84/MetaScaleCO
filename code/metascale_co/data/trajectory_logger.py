"""Logs transitions (s_t, a_t, c_t, G_t, s_{t+1}) as in Eq. (32).
We store the encoded state (full and difficulty-only versions), the action,
the cost in C0 units and the gain in %."""
from __future__ import annotations

import numpy as np

from ..controllers.state_encoder import encode_state


class TrajectoryLogger:
    def __init__(self):
        self.z, self.z_static, self.a, self.c, self.g, self.traj = [], [], [], [], [], []

    def before(self, env):
        self._z = encode_state(env, history=True)
        self._zs = encode_state(env, history=False)

    def after(self, env, a, gain, cost, traj_id):
        self.z.append(self._z); self.z_static.append(self._zs)
        self.a.append(a); self.c.append(cost / env.C0); self.g.append(100.0 * gain)
        self.traj.append(traj_id)

    def arrays(self):
        return dict(z=np.array(self.z, np.float32), z_static=np.array(self.z_static, np.float32),
                    a=np.array(self.a, np.int64), c=np.array(self.c, np.float32),
                    g=np.array(self.g, np.float32), traj=np.array(self.traj, np.int64))
