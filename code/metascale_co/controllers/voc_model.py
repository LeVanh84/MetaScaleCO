"""Value-of-computation model (proposal Sec. 5.4, 6.3).

* Gain head: heteroscedastic Gaussian over the normalised one-step gain G (in %),
  trained with Eq. (33). Its sigma is the *aleatoric* spread of G.
* Cost head: predicts log(c / C0), trained with Eq. (34).
* An ensemble of K independently initialised models gives an *epistemic*
  uncertainty estimate (std of the ensemble means). The proposal's Eq. (27) uses
  sigma_theta from Eq. (33); the epistemic variant is our addition, tested as an
  ablation.

Training uses PyTorch; inference is pure numpy (fast inside the search loop).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

GAIN_CLIP = 20.0   # % ; clip heavy-tailed targets


class _Net(nn.Module):
    def __init__(self, d_in: int, width: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, 3)   # mu, log var, log cost

    def forward(self, x):
        return self.head(self.body(x))


def train_ensemble(X: np.ndarray, gain: np.ndarray, cost_units: np.ndarray,
                   n_models: int = 5, epochs: int = 25, seed: int = 0,
                   batch: int = 1024, lr: float = 2e-3, log=print) -> dict:
    """X: (N, d) state||action features; gain in %; cost in C0 units."""
    torch.set_num_threads(8)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = torch.tensor((X - mean) / std, dtype=torch.float32)
    g = torch.tensor(np.clip(gain, -GAIN_CLIP, GAIN_CLIP), dtype=torch.float32)
    lc = torch.tensor(np.log(np.maximum(cost_units, 1e-4)), dtype=torch.float32)
    N = len(Xn)
    models = []
    for k in range(n_models):
        torch.manual_seed(seed * 100 + k)
        rng = np.random.default_rng(seed * 100 + k)
        idx_boot = rng.integers(0, N, N)            # bootstrap -> diverse ensemble
        net = _Net(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            perm = rng.permutation(idx_boot)
            tot = 0.0
            for s in range(0, N, batch):
                b = perm[s:s + batch]
                out = net(Xn[b])
                mu, logv, lch = out[:, 0], out[:, 1].clamp(-8, 6), out[:, 2]
                l_voc = 0.5 * ((g[b] - mu) ** 2 * torch.exp(-logv) + logv)   # Eq. (33)
                l_cost = (lc[b] - lch).abs()                                  # Eq. (34)
                loss = l_voc.mean() + l_cost.mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            sched.step()
        log(f"  model {k}: final epoch loss {tot / N:.4f}")
        models.append([p.detach().numpy().copy() for p in net.parameters()])
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "models": models}


def save(path, bundle: dict):
    flat = {"mean": bundle["mean"], "std": bundle["std"], "K": len(bundle["models"])}
    for k, ps in enumerate(bundle["models"]):
        for j, p in enumerate(ps):
            flat[f"m{k}_p{j}"] = p
    np.savez(path, **flat)


def _silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyVoC:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std, K = z["mean"], z["std"], int(z["K"])
        self.models = [[z[f"m{k}_p{j}"] for j in range(6)] for k in range(K)]

    def predict(self, X: np.ndarray):
        """Returns mu (%), sigma_aleatoric, sigma_epistemic, predicted cost (C0 units)."""
        Xn = (X - self.mean) / self.std
        mus, vs, lcs = [], [], []
        for W1, b1, W2, b2, W3, b3 in self.models:
            h = _silu(Xn @ W1.T + b1)
            h = _silu(h @ W2.T + b2)
            o = h @ W3.T + b3
            mus.append(o[:, 0]); vs.append(np.exp(np.clip(o[:, 1], -8, 6))); lcs.append(o[:, 2])
        mus, vs, lcs = np.array(mus), np.array(vs), np.array(lcs)
        mu = mus.mean(0)
        s_ale = np.sqrt(vs.mean(0))
        s_epi = mus.std(0)
        cost = np.exp(lcs.mean(0))
        return mu, s_ale, s_epi, cost
