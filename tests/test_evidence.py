import dataclasses

import numpy as np

from dapilot.evidence import (build_evidence_families, conditional_error_covariance,
                              disagreement_distribution, local_residual_reliability,
                              message_influence_stability, _drop_edges)
from dapilot.features import build_selector_features
from dapilot.graphs import STRUCTURAL_NAMES, consensus_graph, structural_diagnostics
from dapilot.splits import TemporalSplits
from dapilot.discovery import _screen_summary


def _predictions(obs, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "gcn": rng.normal(size=obs.returns.shape),
        "mlp": rng.normal(size=obs.returns.shape),
        "placebo_0": rng.normal(size=obs.returns.shape),
        "placebo_1": rng.normal(size=obs.returns.shape),
    }


def test_label_based_evidence_is_invariant_to_future_outcomes(cfg, world):
    obs = world.observable
    splits = TemporalSplits(cfg["splits"], obs.T)
    preds = _predictions(obs)
    t0 = 1500
    f1, _ = local_residual_reliability(obs, preds, splits)
    f2, f2_names = conditional_error_covariance(obs, preds, splits)

    changed = obs.returns.copy()
    changed[t0 + 1:] += 1000
    obs_changed = dataclasses.replace(obs, returns=changed)
    f1_changed, _ = local_residual_reliability(obs_changed, preds, splits)
    f2_changed, _ = conditional_error_covariance(obs_changed, preds, splits)
    np.testing.assert_array_equal(f1[:t0 + 1], f1_changed[:t0 + 1])
    np.testing.assert_array_equal(f2[:t0 + 1], f2_changed[:t0 + 1])

    for j, name in enumerate(f2_names):
        if "mix_weight" in name:
            finite = np.isfinite(f2[:, j])
            assert ((0 <= f2[finite, j]) & (f2[finite, j] <= 1)).all()


def test_prediction_and_message_evidence_is_causal(cfg, world):
    obs = world.observable
    splits = TemporalSplits(cfg["splits"], obs.T)
    preds = _predictions(obs)
    t0 = 1400
    f4, _ = disagreement_distribution(obs, preds, splits)
    changed = {k: v.copy() for k, v in preds.items()}
    changed["gcn"][t0 + 1:] += 500
    f4_changed, _ = disagreement_distribution(obs, changed, splits)
    np.testing.assert_array_equal(f4[:t0 + 1], f4_changed[:t0 + 1])

    rng = np.random.default_rng(8)
    counterfactuals = {
        "base": rng.normal(size=obs.returns.shape),
        "drop20": rng.normal(size=obs.returns.shape),
        "lag1": rng.normal(size=obs.returns.shape),
    }
    f3, _ = message_influence_stability(counterfactuals)
    cf_changed = {k: v.copy() for k, v in counterfactuals.items()}
    cf_changed["drop20"][t0 + 1:] += 500
    f3_changed, _ = message_influence_stability(cf_changed)
    np.testing.assert_array_equal(f3[:t0 + 1], f3_changed[:t0 + 1])

    dropped_1 = _drop_edges(obs.A_obs, 0.20, seed=44)
    dropped_2 = _drop_edges(obs.A_obs, 0.20, seed=44)
    np.testing.assert_array_equal(dropped_1, dropped_2)
    extended = np.concatenate([obs.A_obs, obs.A_obs[-5:]], axis=0)
    np.testing.assert_array_equal(dropped_1, _drop_edges(extended, 0.20, seed=44)[:obs.T])


def test_new_families_are_not_existing_decision_columns(cfg, world):
    obs = world.observable
    splits = TemporalSplits(cfg["splits"], obs.T)
    preds = _predictions(obs, 11)
    train = splits.decision_times("expert_train")
    consensus = consensus_graph(obs.A_obs[train])
    structural = structural_diagnostics(obs.A_obs, obs.X, obs.sectors, consensus)
    base = build_selector_features(
        obs, preds, "gcn", "mlp", ["placebo_0", "placebo_1"], structural,
        STRUCTURAL_NAMES, splits, cfg["selectors"],
    )
    base_cols = base.columns(("evidence", "disagreement", "market"))
    idx = splits.decision_times("meta_train")
    rng = np.random.default_rng(13)
    counterfactuals = {
        "base": rng.normal(size=obs.returns.shape),
        "drop20": rng.normal(size=obs.returns.shape),
        "lag1": rng.normal(size=obs.returns.shape),
    }
    families = build_evidence_families(obs, preds, splits, counterfactuals)
    seen_names = set(base.names)
    for _, (values, names) in families.items():
        assert not seen_names.intersection(names)
        assert np.isfinite(values[idx]).all()
        for j in range(values.shape[1]):
            assert not any(np.allclose(values[idx, j], base.Z[idx, k]) for k in base_cols)
        seen_names.update(names)


def test_oracle_screen_uses_familywise_correction_and_risk(cfg):
    rows = []
    for family_index, family in enumerate(("F1", "F2", "F3", "F4")):
        for seed in range(10):
            diff = -0.02 if family == "F1" else 0.001 * (family_index + 1)
            for subset in ("all", "informative", "no_signal", "corrupted"):
                rows.append({"seed": seed, "family": family, "subset": subset,
                             "risk_diff": diff, "corr_diff": -diff})
    summary = _screen_summary(rows, cfg["evaluation"]["alpha"])
    primary = {r["family"]: r for r in summary
               if r["subset"] == "all" and r["metric"] == "risk_diff"}
    assert primary["F1"]["screening"] == "pass"
    assert 0 <= primary["F1"]["p_holm"] <= 1
    assert all(primary[f]["screening"] == "not_supported" for f in ("F2", "F3", "F4"))
