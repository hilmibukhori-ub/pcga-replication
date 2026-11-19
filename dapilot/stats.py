"""Seed-level paired comparisons with uncertainty intervals and Holm correction."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_summary(diff: np.ndarray, ci: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Summary of per-seed paired differences d_s = metric_A(s) - metric_B(s)."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    tq = stats.t.ppf(0.5 + ci / 2, n - 1) if n > 1 else np.nan
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    lo, hi = np.quantile(boots, [0.5 - ci / 2, 0.5 + ci / 2])
    if n > 1 and np.any(d != 0):
        p = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    else:
        p = 1.0
    return {
        "n": n, "mean": mean, "sd": sd,
        "t_ci_low": mean - tq * se, "t_ci_high": mean + tq * se,
        "boot_ci_low": float(lo), "boot_ci_high": float(hi),
        "wilcoxon_p": p, "n_negative": int((d < 0).sum()), "n_positive": int((d > 0).sum()),
    }


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def verdict(summary: dict, p_adj: float, predicted: str, alpha: float = 0.05) -> str:
    """predicted in {'negative', 'positive', 'two-sided'} — the expected sign of the mean diff.

    supported:     adjusted Wilcoxon p < alpha, mean in predicted direction, and the t-CI
                   excludes zero in that direction.
    contradicted:  same evidence standard, opposite direction.
    inconclusive:  otherwise.
    For 'two-sided' (exploratory) the verdict reports 'difference' / 'no clear difference'.
    """
    m, lo, hi = summary["mean"], summary["t_ci_low"], summary["t_ci_high"]
    sig = p_adj < alpha
    if predicted == "two-sided":
        return "difference detected" if sig and (hi < 0 or lo > 0) else "no clear difference"
    if sig and predicted == "negative" and hi < 0 or sig and predicted == "positive" and lo > 0:
        return "supported"
    if sig and predicted == "negative" and lo > 0 or sig and predicted == "positive" and hi < 0:
        return "contradicted"
    return "inconclusive"
