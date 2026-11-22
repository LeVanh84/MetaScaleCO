"""Process-pool map that degrades to a plain loop when workers == 1
(needed on the GPU server: JAX/CUDA must not be shared across forked workers)."""
from __future__ import annotations

from multiprocessing import Pool


def pmap(fn, tasks, workers: int, initializer=None, initargs=(), chunksize: int = 1):
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(t) for t in tasks]
    with Pool(workers, initializer=initializer, initargs=initargs) as pool:
        return pool.map(fn, tasks, chunksize=chunksize)
