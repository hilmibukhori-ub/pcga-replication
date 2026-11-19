import dataclasses
import copy

import numpy as np

from dapilot.dgp import CORRUPTED, ObservableData, generate_world
from dapilot.graphs import (STRUCTURAL_NAMES, consensus_graph, mean_aggregation, permute_graph,
                            rewire, structural_diagnostics)


def test_shapes_and_determinism(cfg, world):
    o, h = world.observable, world.hidden
    T, N, F = cfg["data"]["n_steps"], cfg["data"]["n_assets"], cfg["data"]["n_features"]
    assert o.X.shape == (T, N, F) and o.A_obs.shape == (T, N, N) and o.returns.shape == (T, N)
    assert set(np.unique(h.regime)) == {0, 1, 2}
    w2 = generate_world(cfg, seed=12345)
    np.testing.assert_array_equal(w2.observable.returns, o.returns)
    np.testing.assert_array_equal(w2.observable.A_obs, o.A_obs)


def test_generated_prefix_is_invariant_to_future_horizon(cfg):
    """Changing how much future is simulated must not alter an already generated prefix."""
    short_cfg, long_cfg = copy.deepcopy(cfg), copy.deepcopy(cfg)
    short_cfg["data"]["n_steps"] = 420
    long_cfg["data"]["n_steps"] = 560
    short = generate_world(short_cfg, seed=2468)
    long = generate_world(long_cfg, seed=2468)
    n = short_cfg["data"]["n_steps"]

    np.testing.assert_array_equal(short.observable.X, long.observable.X[:n])
    np.testing.assert_array_equal(short.observable.factors, long.observable.factors[:n])
    np.testing.assert_array_equal(short.observable.A_obs, long.observable.A_obs[:n])
    np.testing.assert_array_equal(short.observable.returns, long.observable.returns[:n])
    np.testing.assert_array_equal(short.hidden.mu[:-1], long.hidden.mu[:n - 1])
    np.testing.assert_array_equal(short.hidden.regime, long.hidden.regime[:n])


def test_observable_has_no_hidden_fields():
    names = {f.name for f in dataclasses.fields(ObservableData)}
    assert names == {"X", "factors", "A_obs", "sectors", "returns"}
    assert not names & {"A_true", "mu", "regime", "corruption", "segment"}


def test_timestamp_alignment_mu_targets_next_return(world):
    """mu[t] is the conditional mean of returns[t+1]: residuals are mean-zero and
    uncorrelated with mu[t], whereas mu[t] vs returns[t] (misaligned) is not the mean."""
    o, h = world.observable, world.hidden
    res = o.returns[1:] - h.mu[:-1]
    assert abs(res.mean()) < 0.05
    # aligned residual is orthogonal to mu[t]; a one-step misalignment is not
    orth = np.corrcoef(h.mu[:-1].ravel(), res.ravel())[0, 1]
    res_bad = o.returns[1:-1] - h.mu[1:-1]          # treats mu[t] as the mean of returns[t]
    orth_bad = np.corrcoef(h.mu[1:-1].ravel(), res_bad.ravel())[0, 1]
    assert abs(orth) < 0.02 and orth_bad < -0.05
    c_aligned = np.corrcoef(h.mu[:-1].ravel(), o.returns[1:].ravel())[0, 1]
    c_shift = np.corrcoef(h.mu[1:-1].ravel(), o.returns[1:-1].ravel())[0, 1]
    assert c_aligned > c_shift
    assert np.isnan(o.returns[0]).all() and np.isnan(h.mu[-1]).all()


def test_observed_graph_corruption(world):
    o, h = world.observable, world.hidden
    jac = ((o.A_obs & h.A_true).sum((1, 2)) / (o.A_obs | h.A_true).sum((1, 2)))
    assert jac[h.regime != CORRUPTED].mean() > 0.8
    assert jac[h.regime == CORRUPTED].mean() < 0.6


def test_graph_utils():
    rng = np.random.default_rng(0)
    A = rng.random((10, 10)) < 0.3
    A = np.triu(A, 1)
    A = A | A.T
    M = mean_aggregation(A)
    deg = A.sum(1)
    np.testing.assert_allclose(M.sum(1)[deg > 0], 1, atol=1e-6)
    perm = rng.permutation(10)
    P = permute_graph(A, perm)
    assert sorted(P.sum(1)) == sorted(A.sum(1)) and (P == P.T).all()
    R = rewire(A, 0.5, rng)
    assert R.sum() == A.sum() and (R == R.T).all() and not R.diagonal().any()


def test_structural_diagnostics_detect_corruption(world):
    o, h = world.observable, world.hidden
    cons = consensus_graph(o.A_obs[:800])
    S = structural_diagnostics(o.A_obs, o.X, o.sectors, cons)
    assert S.shape == (o.T, len(STRUCTURAL_NAMES)) and np.isfinite(S).all()
    j = STRUCTURAL_NAMES.index("jaccard_consensus")
    assert S[h.regime == CORRUPTED, j].mean() < S[h.regime != CORRUPTED, j].mean() - 0.2
    # diagnostics at t do not depend on the future
    A2 = o.A_obs.copy()
    A2[1500:] = ~A2[1500:]
    S2 = structural_diagnostics(A2, o.X, o.sectors, cons)
    np.testing.assert_array_equal(S[:1500], S2[:1500])
