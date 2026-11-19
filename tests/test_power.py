import numpy as np

from dapilot.power import (adjacency, block_bootstrap_lower, degree_edges, industry_edges,
                           same_industry_counts, sample_graph)


def _g(seed=0, n=200):
    rng = np.random.default_rng(seed)
    E, blocks = sample_graph(rng, n, 10, 3.0, 3.0)
    return E, blocks, rng, n


def _deg(E, n):
    return np.bincount(E.ravel(), minlength=n)


def _simple(E):
    assert (E[:, 0] < E[:, 1]).all() and len({tuple(e) for e in E.tolist()}) == len(E)


def test_degree_swap_preserves_degrees():
    E, _, rng, n = _g()
    E2 = degree_edges(E, rng)
    _simple(E2)
    np.testing.assert_array_equal(_deg(E2, n), _deg(E, n))
    overlap = len({tuple(e) for e in E2.tolist()} & {tuple(e) for e in E.tolist()}) / len(E)
    assert overlap < 0.3


def test_industry_swap_preserves_degree_and_same_industry_counts():
    E, blocks, rng, n = _g(1)
    E3 = industry_edges(E, blocks, rng)
    _simple(E3)
    np.testing.assert_array_equal(_deg(E3, n), _deg(E, n))
    np.testing.assert_array_equal(same_industry_counts(E3, blocks, n), same_industry_counts(E, blocks, n))
    inter = {tuple(e) for e in E.tolist() if blocks[e[0]] != blocks[e[1]]}
    inter3 = {tuple(e) for e in E3.tolist() if blocks[e[0]] != blocks[e[1]]}
    assert len(inter & inter3) / len(inter) < 0.2      # inter-industry edges really are randomised


def test_adjacency_symmetric():
    E, _, _, n = _g(2)
    A = adjacency(E, n)
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0


def test_block_bootstrap_detects_shift_and_not_null():
    rng = np.random.default_rng(0)
    d = np.stack([rng.normal(0, 1, 2000), rng.normal(0.3, 1, 2000)], 1)
    _, lo, hi = block_bootstrap_lower(d, 20, 500, 0.95, rng)
    assert lo[0] < 0 < hi[0] and lo[1] > 0


def test_rank_test_p_value():
    from dapilot.power_v2 import rank_test_p
    l3 = np.arange(1, 20, dtype=float)[:, None] * np.ones((19, 2))
    p = rank_test_p(np.array([0.5, 25.0]), l3)
    np.testing.assert_allclose(p, [1 / 20, 1.0])
