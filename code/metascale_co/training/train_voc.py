"""Stage 2 (Sec. 6.3): supervised VoC pretraining on the trajectory bank.
Trains two ensembles: the full model (instance + history + budget) and the
difficulty-only ablation (instance + budget, no history).

Usage:  uv run python -m metascale_co.training.train_voc
"""
from __future__ import annotations

import numpy as np

from ..config import ART, load_config
from ..controllers import voc_model
from ..operators.classical import OP_NAMES


def main():
    cfg = load_config()["voc"]
    bank = dict(np.load(ART / "trajectory_bank.npz"))
    A = len(OP_NAMES)
    onehot = np.eye(A, dtype=np.float32)[bank["a"]]
    # hold out 10 % of trajectories for a calibration sanity check
    rng = np.random.default_rng(0)
    trajs = np.unique(bank["traj"])
    held = set(rng.choice(trajs, len(trajs) // 10, replace=False).tolist())
    is_held = np.array([t in held for t in bank["traj"]])
    for tag, key in [("full", "z"), ("static", "z_static")]:
        X = np.concatenate([bank[key], onehot], 1)
        print(f"training VoC ensemble [{tag}] on {(~is_held).sum()} transitions, d={X.shape[1]}")
        bundle = voc_model.train_ensemble(X[~is_held], bank["g"][~is_held], bank["c"][~is_held],
                                          n_models=cfg["n_models"], epochs=cfg["epochs"])
        path = ART / "models" / f"voc_{tag}.npz"
        voc_model.save(path, bundle)
        voc = voc_model.NumpyVoC(path)
        mu, s_ale, s_epi, cost = voc.predict(X[is_held])
        g, c = bank["g"][is_held], bank["c"][is_held]
        rho = np.corrcoef(mu, g)[0, 1]
        mae_c = np.mean(np.abs(np.log(cost) - np.log(np.maximum(c, 1e-4))))
        print(f"  held-out: corr(mu, G)={rho:.3f}  MAE gain={np.mean(np.abs(mu - g)):.3f}%  "
              f"MAE log-cost={mae_c:.3f}  mean sigma_ale={s_ale.mean():.3f} sigma_epi={s_epi.mean():.3f}")


if __name__ == "__main__":
    main()
