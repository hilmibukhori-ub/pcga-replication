import numpy as np
import pandas as pd

from dapilot.diagnostics import _ridge_benefit, tune_soft_temperature
from dapilot.ceiling import feature_groups, optimal_soft_weight
from dapilot.metrics import calibration_stats, evaluate_selectors
from dapilot.report import hypothesis_diffs
from dapilot.selectors import placebo_calibrated_weight
from dapilot.stats import holm, paired_summary, verdict


def _toy(T=60, N=5, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(T, N))
    pG = mu + rng.normal(scale=0.5, size=(T, N))
    pN = mu + rng.normal(scale=0.5, size=(T, N))
    y = mu + rng.normal(size=(T, N))
    regime = rng.integers(0, 3, size=T)
    return mu, pG, pN, y, regime


def test_regret_properties():
    mu, pG, pN, y, regime = _toy()
    idx = np.arange(60)
    RG, RN = ((mu - pG) ** 2).mean(1), ((mu - pN) ** 2).mean(1)
    outs = {
        "G": {"w": np.ones(60), "benefit": None, "d2": None},
        "N": {"w": np.zeros(60), "benefit": None, "d2": None},
        "perfect": {"w": (RG < RN).astype(float), "benefit": RN - RG, "d2": None},
    }
    r = evaluate_selectors(outs, pG, pN, y, mu, regime, idx)
    assert abs(r["Oracle"]["all"]["regret"]) < 1e-12
    assert abs(r["perfect"]["all"]["regret"]) < 1e-12
    assert r["perfect"]["all"]["benefit_rmse"] < 1e-12
    assert abs(r["perfect"]["all"]["calib_slope"] - 1) < 1e-8
    assert r["G"]["all"]["regret"] >= 0 and r["N"]["all"]["regret"] >= 0
    assert r["perfect"]["all"]["oracle_agreement"] == 1.0
    np.testing.assert_allclose(r["G"]["all"]["regret"], (RG - np.minimum(RG, RN)).mean())


def test_calibration_stats_identity():
    x = np.linspace(-1, 1, 200)
    c = calibration_stats(x, x, 10)
    assert abs(c["slope"] - 1) < 1e-9 and abs(c["intercept"]) < 1e-9 and c["ece"] < 1e-9


def test_pc_sgl_weight_requires_both_no_graph_and_placebo_evidence():
    benefit = np.array([2.0, 2.0, -2.0, -2.0, 0.0])
    d2 = np.array([2.0, -2.0, 2.0, -2.0, 0.0])
    w = placebo_calibrated_weight(benefit, d2, 1.0, 1.0, 0.25)
    assert w[0] > 0.99
    assert w[1] < 0.01 and w[2] < 0.01 and w[3] < 0.01
    assert w[4] == 0.5


def test_pc_sgl_weight_is_scale_invariant():
    b = np.array([-1.0, 0.5, 2.0])
    d = np.array([1.0, 0.25, 4.0])
    w1 = placebo_calibrated_weight(b, d, 2.0, 4.0, 0.5)
    w2 = placebo_calibrated_weight(10 * b, 3 * d, 20.0, 12.0, 0.5)
    np.testing.assert_allclose(w1, w2)


def test_paired_summary_and_holm():
    d = np.array([-1.0, -0.8, -1.2, -0.9, -1.1, -1.0, -0.7, -1.3, -0.95, -1.05])
    s = paired_summary(d, n_boot=2000)
    assert s["t_ci_high"] < 0 and s["wilcoxon_p"] < 0.01 and s["n_negative"] == 10
    assert verdict(s, s["wilcoxon_p"], "negative") == "supported"
    assert verdict(s, s["wilcoxon_p"], "positive") == "contradicted"
    noisy = paired_summary(np.array([1, -1, 0.5, -0.5, 0.2, -0.3, 0.1, -0.1, 0.4, -0.2]), n_boot=500)
    assert verdict(noisy, noisy["wilcoxon_p"], "negative") == "inconclusive"
    adj = holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_ha6_separates_expert_quality_from_selector_gain():
    rows = []
    # Real graph expert is much better in absolute terms, but A2a adds +2 over AlwaysG in both
    # slots.  HA6 must therefore be zero rather than rediscovering the expert-quality gap.
    risks = {
        ("real", "AlwaysG"): 4.0,
        ("real", "A2a"): 2.0,
        ("placebo", "AlwaysG"): 10.0,
        ("placebo", "A2a"): 8.0,
    }
    for seed in (0, 1):
        for (slot, method), value in risks.items():
            rows.append({
                "seed": seed, "table": "selector", "slot": slot, "method": method,
                "subset": "all", "metric": "cond_risk", "value": value,
            })
    df = pd.DataFrame(rows)
    expert_gap = hypothesis_diffs(df, "expert_slot", ("all",))
    selector_gain = hypothesis_diffs(df, "selector_gain", ("A2a", "AlwaysG", "all"))
    np.testing.assert_allclose(expert_gap, [6.0, 6.0])
    np.testing.assert_allclose(selector_gain, [0.0, 0.0])


def test_diagnostics_ignore_unavailable_early_rows():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(30, 4))
    z[:3] = np.nan
    target = rng.normal(size=30)
    fit, val, cal, test = np.arange(3, 12), np.arange(12, 18), np.arange(18, 24), np.arange(24, 30)
    pred, _ = _ridge_benefit(
        z, target, fit, val, cal, test, np.arange(4), [0.1, 1.0], "affine"
    )
    assert pred.shape == (len(test),) and np.isfinite(pred).all()

    p_g, p_n, y = rng.normal(size=(3, 30, 2))
    benefit = rng.normal(size=30)
    temperature, losses = tune_soft_temperature(benefit, p_g, p_n, y, cal)
    assert temperature > 0 and np.isfinite(losses).all()


def test_ceiling_soft_oracle_and_feature_groups():
    rng = np.random.default_rng(7)
    mu = rng.normal(size=(40, 6))
    p_g = rng.normal(size=(40, 6))
    p_n = rng.normal(size=(40, 6))
    w = optimal_soft_weight(mu, p_g, p_n)
    soft = ((mu - (w[:, None] * p_g + (1 - w[:, None]) * p_n)) ** 2).mean(1)
    hard = np.minimum(((mu - p_g) ** 2).mean(1), ((mu - p_n) ** 2).mean(1))
    assert ((0 <= w) & (w <= 1)).all()
    assert np.all(soft <= hard + 1e-12)

    names = np.array(["S_density", "E_benefit", "Q_rms", "M_factor"])
    groups = feature_groups(names)
    np.testing.assert_array_equal(groups["structural"], [0])
    np.testing.assert_array_equal(groups["history"], [1])
    np.testing.assert_array_equal(groups["decision"], [1, 2, 3])
    np.testing.assert_array_equal(groups["all"], [0, 1, 2, 3])
