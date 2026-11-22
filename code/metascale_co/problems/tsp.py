"""Euclidean TSP: instance generation and numba search kernels.

Compute is measured in *move evaluations* (``evals``): every delta-cost evaluation
of a 2-opt / or-opt move and every insertion-position evaluation counts as one.
This is the CO analogue of function evaluations (FEs): hardware independent and
identical for every method. Every kernel accepts ``max_evals`` and never exceeds it,
so a run can never overspend its budget.
"""
from __future__ import annotations

import numpy as np
from numba import njit

EPS = 1e-10
K_NEIGH = 10


# --------------------------------------------------------------------------- instances
def uniform_instance(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, 2))


def clustered_instance(n: int, rng: np.random.Generator, n_clusters: int = 5,
                       std: float = 0.06) -> np.ndarray:
    """Gaussian clusters inside the unit square (geometry shift for OOD tests)."""
    centers = rng.random((n_clusters, 2)) * 0.8 + 0.1
    lab = rng.integers(0, n_clusters, size=n)
    pts = centers[lab] + rng.normal(0.0, std, size=(n, 2))
    return np.clip(pts, 0.0, 1.0)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def neighbor_lists(D: np.ndarray, k: int = K_NEIGH) -> np.ndarray:
    order = np.argsort(D, axis=1)
    return np.ascontiguousarray(order[:, 1:k + 1]).astype(np.int64)


class TSPInstance:
    """Holds coordinates, distance matrix and candidate neighbour lists."""

    def __init__(self, coords: np.ndarray, name: str = ""):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = len(coords)
        self.D = distance_matrix(self.coords)
        self.neigh = neighbor_lists(self.D, min(K_NEIGH, self.n - 1))
        self.name = name
        # static descriptors used by the state encoder (dimensionless)
        nn = np.sort(self.D, axis=1)[:, 1]
        self.mean_nn = float(nn.mean() * np.sqrt(self.n))     # ~0.5 for uniform points
        self.cv_nn = float(nn.std() / (nn.mean() + 1e-12))       # high for clustered points


# --------------------------------------------------------------------------- kernels
@njit(cache=True)
def tour_length(tour, D):
    n = tour.shape[0]
    s = 0.0
    for i in range(n - 1):
        s += D[tour[i], tour[i + 1]]
    return s + D[tour[n - 1], tour[0]]


@njit(cache=True)
def _reverse(tour, pos, p, q):
    """Reverse the circular segment tour[p..q] (forward direction, inclusive)."""
    n = tour.shape[0]
    length = (q - p) % n + 1
    for k in range(length // 2):
        i = (p + k) % n
        j = (q - k) % n
        a = tour[i]
        b = tour[j]
        tour[i] = b
        tour[j] = a
        pos[b] = i
        pos[a] = j


@njit(cache=True)
def two_opt(tour, D, neigh, max_evals):
    """First-improvement 2-opt with neighbour lists (successor + predecessor moves).

    Runs to a local optimum or until ``max_evals`` move evaluations are used.
    Returns (tour, evals_used). Operates on a copy.
    """
    n = tour.shape[0]
    t = tour.copy()
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[t[i]] = i
    evals = 0
    improved = True
    while improved:
        improved = False
        for i in range(n):
            a = t[i]
            # successor move: edges (a,b),(c,d) -> (a,c),(b,d)
            b = t[(i + 1) % n]
            dab = D[a, b]
            for kk in range(neigh.shape[1]):
                c = neigh[a, kk]
                if D[a, c] >= dab:
                    break
                if evals >= max_evals:
                    return t, evals
                evals += 1
                j = pos[c]
                d = t[(j + 1) % n]
                if c == b or d == a:
                    continue
                delta = D[a, c] + D[b, d] - dab - D[c, d]
                if delta < -EPS:
                    _reverse(t, pos, (i + 1) % n, j)
                    improved = True
                    break
            if improved:
                break
            # predecessor move: edges (b,a),(d,c) -> (c,a),(d,b)
            b = t[(i - 1) % n]
            dab = D[a, b]
            for kk in range(neigh.shape[1]):
                c = neigh[a, kk]
                if D[a, c] >= dab:
                    break
                if evals >= max_evals:
                    return t, evals
                evals += 1
                j = pos[c]
                d = t[(j - 1) % n]
                if c == b or d == a:
                    continue
                delta = D[a, c] + D[b, d] - dab - D[c, d]
                if delta < -EPS:
                    _reverse(t, pos, i, (j - 1) % n)
                    improved = True
                    break
            if improved:
                break
    return t, evals


@njit(cache=True)
def or_opt(tour, D, neigh, max_evals):
    """Or-opt: move a segment of 1-3 consecutive cities next to a candidate neighbour.

    First improvement, repeated to a local optimum or until the eval cap.
    """
    n = tour.shape[0]
    t = tour.copy()
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[t[i]] = i
    evals = 0
    improved = True
    while improved:
        improved = False
        for L in range(1, 4):
            for i in range(n):
                s1 = t[i]
                sL = t[(i + L - 1) % n]
                p = t[(i - 1) % n]
                nx = t[(i + L) % n]
                rem_gain = D[p, s1] + D[sL, nx] - D[p, nx]
                for end in range(2):
                    head = s1 if end == 0 else sL
                    for kk in range(neigh.shape[1]):
                        c = neigh[head, kk]
                        # c must lie outside the segment
                        off = (pos[c] - i) % n
                        if off < L:
                            continue
                        e = t[(pos[c] + 1) % n]
                        if (pos[e] - i) % n < L:
                            continue
                        if evals >= max_evals:
                            return t, evals
                        evals += 1
                        # insert between c and e; orientation chosen so head is next to c
                        if end == 0:
                            add = D[c, s1] + D[sL, e] - D[c, e]
                        else:
                            add = D[c, sL] + D[s1, e] - D[c, e]
                        if add - rem_gain < -EPS:
                            # rebuild tour: remove segment, insert after c
                            seg = np.empty(L, np.int64)
                            for q in range(L):
                                seg[q] = t[(i + q) % n]
                            if end == 1:
                                seg = seg[::-1].copy()
                            rest = np.empty(n - L, np.int64)
                            r = 0
                            for q in range(n):
                                v = t[(i + L + q) % n]
                                if (pos[v] - i) % n >= L:
                                    rest[r] = v
                                    r += 1
                            newt = np.empty(n, np.int64)
                            w = 0
                            for q in range(n - L):
                                newt[w] = rest[q]
                                w += 1
                                if rest[q] == c:
                                    for z in range(L):
                                        newt[w] = seg[z]
                                        w += 1
                            t = newt
                            for q in range(n):
                                pos[t[q]] = q
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return t, evals


@njit(cache=True)
def destroy_repair(tour, D, neigh, ratio, seed, max_evals):
    """Remove a spatially clustered fraction ``ratio`` of cities, reinsert them by
    cheapest insertion in random order, then 2-opt. Analogue of a GenSCO
    disruption-regeneration cycle (local disruption + reconstruction).
    Returns (candidate, evals, ok). ok=False if the budget ran out mid-repair.
    """
    np.random.seed(seed)
    n = tour.shape[0]
    m = max(3, int(round(ratio * n)))
    center = np.random.randint(n)
    dist_c = D[center].copy()
    order = np.argsort(dist_c)
    removed = np.zeros(n, np.bool_)
    rem = order[:m].copy()
    for v in rem:
        removed[v] = True
    partial = np.empty(n, np.int64)
    L = 0
    for q in range(n):
        if not removed[tour[q]]:
            partial[L] = tour[q]
            L += 1
    np.random.shuffle(rem)
    evals = 0
    for v in rem:
        best = 1e18
        bpos = 0
        for q in range(L):
            if evals >= max_evals:
                return tour.copy(), evals, False
            evals += 1
            a = partial[q]
            b = partial[(q + 1) % L]
            ins = D[a, v] + D[v, b] - D[a, b]
            if ins < best:
                best = ins
                bpos = q
        for q in range(L, bpos + 1, -1):
            partial[q] = partial[q - 1]
        partial[bpos + 1] = v
        L += 1
    cand, e2 = two_opt(partial, D, neigh, max_evals - evals)
    return cand, evals + e2, True


@njit(cache=True)
def double_bridge(tour, seed, n_kicks):
    """Random double-bridge kick(s): a global, non-sequential 4-opt perturbation."""
    np.random.seed(seed)
    n = tour.shape[0]
    t = tour.copy()
    for _ in range(n_kicks):
        cuts = np.sort(np.random.choice(np.arange(1, n), 3, replace=False))
        a, b, c = cuts[0], cuts[1], cuts[2]
        newt = np.empty(n, np.int64)
        w = 0
        for q in range(0, a):
            newt[w] = t[q]; w += 1
        for q in range(c, n):
            newt[w] = t[q]; w += 1
        for q in range(b, c):
            newt[w] = t[q]; w += 1
        for q in range(a, b):
            newt[w] = t[q]; w += 1
        t = newt
    return t


@njit(cache=True)
def nearest_neighbor_tour(D, start):
    n = D.shape[0]
    visited = np.zeros(n, np.bool_)
    t = np.empty(n, np.int64)
    t[0] = start
    visited[start] = True
    for k in range(1, n):
        last = t[k - 1]
        best = 1e18
        bj = -1
        for j in range(n):
            if not visited[j] and D[last, j] < best:
                best = D[last, j]
                bj = j
        t[k] = bj
        visited[bj] = True
    return t


@njit(cache=True)
def random_insertion_tour(D, seed):
    """Random-order cheapest insertion; returns (tour, evals)."""
    np.random.seed(seed)
    n = D.shape[0]
    perm = np.random.permutation(n)
    t = np.empty(n, np.int64)
    t[0] = perm[0]; t[1] = perm[1]; t[2] = perm[2]
    L = 3
    evals = 0
    for k in range(3, n):
        v = perm[k]
        best = 1e18
        bpos = 0
        for q in range(L):
            evals += 1
            a = t[q]; b = t[(q + 1) % L]
            ins = D[a, v] + D[v, b] - D[a, b]
            if ins < best:
                best = ins; bpos = q
        for q in range(L, bpos + 1, -1):
            t[q] = t[q - 1]
        t[bpos + 1] = v
        L += 1
    return t, evals


def is_valid_tour(tour: np.ndarray, n: int) -> bool:
    return len(tour) == n and np.array_equal(np.sort(tour), np.arange(n))
