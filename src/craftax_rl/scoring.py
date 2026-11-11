import numpy as np


def summarize(scores):
    """Mean and 95% bootstrap CI over episodes."""
    a = np.asarray(scores, dtype=float)
    boot = a[np.random.default_rng(0).integers(0, len(a), (2000, len(a)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi)
