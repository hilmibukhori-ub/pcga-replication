"""Candidate D1 tests for models with training (seed) noise. Used by the dev rehearsal to pick a
calibrated test BEFORE the replication pre-registration is frozen.

A, B: (S, T) daily top-1 returns (seeds x test days). The statistic is the IRR difference
D = sum_t (mean_s A[s,t] - mean_s B[s,t]). One-sided H1: D > 0.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def day_bootstrap(A, B, L, R, rng):
    """The method in PREREG v0.1: circular block bootstrap over days of the seed-averaged series."""
    d = A.mean(0) - B.mean(0)
    T = len(d)
    ext = np.concatenate([d, d[:L - 1]])
    cs = np.concatenate([[0.0], np.cumsum(ext)])
    bm = (cs[L:L + T] - cs[:T]) / L
    boots = bm[rng.integers(0, T, (R, max(1, round(T / L))))].mean(1) * T
    return float((boots <= 0).mean())


def seed_t(A, B, paired):
    """t-test on per-seed IRR. paired=True: seed s of A with seed s of B (common random numbers)."""
    a, b = A.sum(1), B.sum(1)
    if paired:
        t, p = stats.ttest_rel(a, b, alternative="greater")
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    return float(p)


def twoway_bootstrap(A, B, L, R, rng, paired):
    """Resample seeds (pairs if paired) and circular day blocks jointly."""
    Sa, T = A.shape
    Sb = B.shape[0]
    nb = max(1, round(T / L))
    out = np.empty(R)
    for r in range(R):
        sa = rng.integers(0, Sa, Sa)
        sb = sa if paired else rng.integers(0, Sb, Sb)
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(L)[None]) % T
        idx = idx.ravel()
        out[r] = (A[sa][:, idx].mean(0) - B[sb][:, idx].mean(0)).mean() * T
    return float((out <= 0).mean())
