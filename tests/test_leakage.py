"""Information-availability tests: no future labels, no hidden variables, locked test."""

import dataclasses

import numpy as np
import pytest
import torch

from dapilot.experts import GraphNet, build_inputs, label_array
from dapilot.features import (build_selector_features, make_meta_data, matured_window_mean)
from dapilot.graphs import STRUCTURAL_NAMES, consensus_graph, structural_diagnostics
from dapilot.splits import LeakageError, TemporalSplits


def _fake_preds(obs, rng):
    T, N = obs.returns.shape
    keys = ["gcn", "mlp", "placebo_0", "placebo_1"]
    p = {k: rng.normal(size=(T, N)).astype(np.float32) for k in keys}
    for k in keys:
        p[k][-1] = np.nan
    return p


def _features(obs, preds, sp, cfg):
    cons = consensus_graph(obs.A_obs[sp.decision_times("expert_train")])
    S = structural_diagnostics(obs.A_obs, obs.X, obs.sectors, cons)
    return build_selector_features(obs, preds, "gcn", "mlp", ["placebo_0", "placebo_1"], S,
                                   STRUCTURAL_NAMES, sp, cfg["selectors"])


def test_label_array_alignment(world):
    y = label_array(world.observable, 1)
    np.testing.assert_array_equal(y[:-1], world.observable.returns[1:].astype(np.float32))
    assert np.isnan(y[-1]).all()


def test_matured_window_never_uses_unmatured_labels():
    s = np.arange(100, dtype=float)
    # at t=50 with h=1 the most recent usable decision time is 49
    assert matured_window_mean(s, 50, 1, 1) == 49.0
    assert matured_window_mean(s, 50, 5, 1) == np.mean(np.arange(45, 50))
    assert np.isnan(matured_window_mean(s, 3, 10, 1))


def test_selector_features_invariant_to_future_returns(cfg, world):
    """Perturbing returns after time t0 must leave every selector feature at times <= t0 unchanged."""
    obs = world.observable
    sp = TemporalSplits(cfg["splits"], obs.T)
    preds = _fake_preds(obs, np.random.default_rng(0))
    F1 = _features(obs, preds, sp, cfg)
    t0 = 1500
    r2 = obs.returns.copy()
    r2[t0 + 1:] += 100.0
    obs2 = dataclasses.replace(obs, returns=r2)
    F2 = _features(obs2, preds, sp, cfg)
    np.testing.assert_array_equal(F1.Z[:t0 + 1], F2.Z[:t0 + 1])
    assert not np.allclose(F1.Z[t0 + 2:], F2.Z[t0 + 2:], equal_nan=True)


def test_selector_features_invariant_to_future_predictions(cfg, world):
    obs = world.observable
    sp = TemporalSplits(cfg["splits"], obs.T)
    preds = _fake_preds(obs, np.random.default_rng(0))
    F1 = _features(obs, preds, sp, cfg)
    p2 = {k: v.copy() for k, v in preds.items()}
    for v in p2.values():
        v[1201:-1] += 5.0
    F2 = _features(obs, p2, sp, cfg)
    np.testing.assert_array_equal(F1.Z[:1201], F2.Z[:1201])


def test_no_in_sample_expert_losses_in_features(cfg, world):
    """Evidence windows must not reach into the expert-train period (in-sample predictions)."""
    obs = world.observable
    sp = TemporalSplits(cfg["splits"], obs.T)
    F = _features(obs, _fake_preds(obs, np.random.default_rng(0)), sp, cfg)
    ev = F.groups["evidence"]
    assert np.isnan(F.Z[: sp.start("expert_val") + 1, ev]).all()
    assert np.isfinite(F.Z[sp.decision_times("meta_train")]).all()


def test_meta_data_has_no_locked_test_labels(cfg, world):
    obs = world.observable
    sp = TemporalSplits(cfg["splits"], obs.T)
    preds = _fake_preds(obs, np.random.default_rng(0))
    md = make_meta_data(obs, preds, "gcn", "mlp", ["placebo_0", "placebo_1"],
                        _features(obs, preds, sp, cfg), sp)
    first_bad = sp.start("test") - sp.horizon       # label matures at test start
    assert np.isnan(md.y[first_bad:]).all()
    assert np.isnan(md.B[first_bad:]).all() and np.isnan(md.LG[first_bad:]).all()
    assert np.isfinite(md.B[md.cal_idx]).all()
    assert md.cal_idx.max() + sp.horizon < sp.start("test")


def test_make_meta_data_rejects_bad_splits(cfg, world):
    """If meta-training started inside expert-val, evidence windows would be undefined."""
    obs = world.observable
    bad = dict(cfg["splits"])
    bad["expert_val"] = [801, 810]
    bad["meta_train"] = [811, 1700]
    sp = TemporalSplits(bad, obs.T)
    preds = _fake_preds(obs, np.random.default_rng(0))
    with pytest.raises(LeakageError):
        make_meta_data(obs, preds, "gcn", "mlp", ["placebo_0"], _features(obs, preds, sp, cfg), sp)


def test_expert_inputs_use_observables_only(world):
    """Expert inputs are a function of ObservableData; altering hidden data changes nothing,
    and inputs at t do not depend on returns at all (labels enter only as targets)."""
    obs = world.observable
    Z = build_inputs(obs, "cross")
    obs2 = dataclasses.replace(obs, returns=obs.returns * 0 + 7.0)
    np.testing.assert_array_equal(Z, build_inputs(obs2, "cross"))


def test_graphnet_permutation_equivariant_and_uses_graph():
    torch.manual_seed(0)
    N, d = 12, 5
    Z = torch.randn(2, N, d)
    A = (torch.rand(2, N, N) < 0.3).float()
    A = ((A + A.transpose(1, 2)) > 0).float()
    A = A / A.sum(-1, keepdim=True).clamp(min=1)
    net = GraphNet(d, 8, True).eval()
    perm = torch.randperm(N)
    out = net(Z, A)
    out_p = net(Z[:, perm], A[:, perm][:, :, perm])
    torch.testing.assert_close(out[:, perm], out_p)
    assert not torch.allclose(out, net(Z, torch.zeros_like(A)))
