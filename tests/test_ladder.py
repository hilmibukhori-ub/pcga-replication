import numpy as np

from dapilot.ladder import block_counts, double_edge_swap, er_graph, sample_graph


def _g(seed=0):
    rng = np.random.default_rng(seed)
    sectors = np.repeat(np.arange(4), 15)
    return sample_graph(rng, sectors, 0.3, 0.02), sectors, rng


def _valid(B):
    assert (B == B.T).all() and not B.diagonal().any()


def test_er_keeps_edge_count():
    A, _, rng = _g()
    B = er_graph(A, rng)
    _valid(B)
    assert B.sum() == A.sum()


def test_degree_swap_keeps_every_degree_and_changes_edges():
    A, _, rng = _g(1)
    B = double_edge_swap(A, rng, 10 * int(A.sum() // 2))
    _valid(B)
    np.testing.assert_array_equal(B.sum(1), A.sum(1))
    assert (B & A).sum() < 0.5 * A.sum()


def test_block_swap_keeps_per_node_block_counts():
    A, s, rng = _g(2)
    B = double_edge_swap(A, rng, 10 * int(A.sum() // 2), blocks=s)
    _valid(B)
    np.testing.assert_array_equal(block_counts(B, s), block_counts(A, s))
    assert (B & A).sum() < 0.8 * A.sum()
