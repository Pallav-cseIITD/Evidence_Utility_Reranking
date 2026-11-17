import math

import numpy as np


def pair(candidate, base, sampled):
    delta = np.asarray(candidate, dtype=int) - np.asarray(base, dtype=int)
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)
    interval = np.quantile(delta[sampled].mean(axis=1) * 100, [0.025, 0.975])
    return dict(wins=wins, losses=losses, ties=len(delta)-n, gain_percentage_points=float(delta.mean()*100),
                bootstrap95_percentage_points=interval.tolist(), exact_paired_p=p)


def paired_binary(left, right):
    """Question-paired exact McNemar and percentile bootstrap; failures count zero."""
    a, b = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Expected aligned nonempty outcomes")
    delta = a - b
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses)+1)) / 2**n) if n else 1.
    rng = np.random.default_rng(20261117)
    values = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1) * 100
    return dict(n=len(a), wins=wins, losses=losses, ties=int((delta == 0).sum()),
                difference_pp=float(delta.mean()*100), bootstrap_95_pp=np.quantile(values, [.025, .975]).tolist(),
                exact_mcnemar_two_sided=p, bootstrap_samples=10000, bootstrap_seed=20261117)
