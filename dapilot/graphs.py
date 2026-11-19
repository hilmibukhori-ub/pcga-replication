"""Graph utilities: aggregation operators, placebo graphs, structural diagnostics."""

from __future__ import annotations

import numpy as np

STRUCTURAL_NAMES = (
    "density",
    "intra_sector_frac",
    "jaccard_prev",
    "jaccard_lag10",
    "jaccard_consensus",
    "feature_homophily",
    "degree_cv",
)


def symmetric_from_upper(vec: np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=bool)
    iu = np.triu_indices(n, 1)
    A[iu] = vec
    return A | A.T


def mean_aggregation(A: np.ndarray) -> np.ndarray:
    """Row-normalised adjacency D^{-1} A (no self loops); isolated nodes get a zero row."""
    A = A.astype(np.float32)
    deg = A.sum(-1, keepdims=True)
    return A / np.maximum(deg, 1.0)


def permute_graph(A: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Placebo graph: A_p[i, j] = A[perm[i], perm[j]]. Preserves density and degree multiset."""
    return A[..., perm, :][..., :, perm]


def rewire(A: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Replace `frac` of the edges of a symmetric graph by uniformly random non-edges."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    vec = A[iu].copy()
    edges, non = np.flatnonzero(vec), np.flatnonzero(~vec)
    k = int(round(frac * len(edges)))
    k = min(k, len(non))
    vec[rng.choice(edges, k, replace=False)] = False
    vec[rng.choice(non, k, replace=False)] = True
    return symmetric_from_upper(vec, n)


def consensus_graph(A_window: np.ndarray) -> np.ndarray:
    """Edges present in more than half of the observed graphs of a window."""
    return A_window.mean(0) > 0.5


def _jaccard(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    inter = (A & B).sum((-2, -1))
    union = (A | B).sum((-2, -1))
    return inter / np.maximum(union, 1)


def structural_diagnostics(
    A: np.ndarray, X: np.ndarray, sectors: np.ndarray, consensus: np.ndarray, lag: int = 10
) -> np.ndarray:
    """Per-time structural diagnostics of the *observed* graph.

    Row t depends only on A[t], A[t-1], A[t-lag], X[t] and a fixed consensus graph, i.e. on
    information available at time t. Returns (T, len(STRUCTURAL_NAMES)).
    """
    T, n, _ = A.shape
    same = sectors[:, None] == sectors[None, :]
    n_pairs = n * (n - 1) / 2
    n_edges = A.sum((1, 2)) / 2
    density = n_edges / n_pairs
    intra = (A & same).sum((1, 2)) / 2 / np.maximum(n_edges, 1)

    prev = np.concatenate([A[:1], A[:-1]], 0)
    lagged = np.concatenate([np.repeat(A[:1], lag, 0), A[:-lag]], 0)
    jac_prev = _jaccard(A, prev)
    jac_lag = _jaccard(A, lagged)
    jac_cons = _jaccard(A, consensus[None])

    M = np.einsum("tij,tjf->tif", mean_aggregation(A), X)
    Xc = X - X.mean(1, keepdims=True)
    Mc = M - M.mean(1, keepdims=True)
    corr = (Xc * Mc).sum(1) / np.sqrt((Xc**2).sum(1) * (Mc**2).sum(1) + 1e-12)
    homophily = corr.mean(-1)

    deg = A.sum(-1)
    degree_cv = deg.std(-1) / np.maximum(deg.mean(-1), 1e-9)

    out = np.stack([density, intra, jac_prev, jac_lag, jac_cons, homophily, degree_cv], -1)
    return out.astype(np.float64)
