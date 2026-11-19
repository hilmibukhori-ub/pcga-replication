import numpy as np
import pandas as pd
import pytest

from dapilot.pcga_l import (
    STEPS,
    decompose_loss_levels,
    positive_step_shares,
    recovery_summary,
    telescoping_error,
)


def _levels(rows):
    return pd.DataFrame(
        rows,
        columns=["L0_none", "L1_er", "L2_degree", "L3_block", "L4_real"],
    )


def test_decomposition_telescopes_and_allows_negative_steps():
    df = decompose_loss_levels(_levels([[10.0, 8.0, 9.0, 6.0, 7.0]]))
    np.testing.assert_allclose(df[list(STEPS)].iloc[0], [2.0, -1.0, 3.0, -1.0])
    assert df.loc[0, "D1"] == 3.0
    np.testing.assert_allclose(telescoping_error(df), 0.0, atol=1e-15)


def test_positive_shares_are_descriptive_and_zero_row_is_undefined():
    df = pd.DataFrame(
        {"density": [2.0, 0.0], "degree": [-1.0, 0.0], "block": [3.0, 0.0], "edge": [-1.0, 0.0]}
    )
    shares = positive_step_shares(df)
    np.testing.assert_allclose(shares.iloc[0], [0.4, 0.0, 0.6, 0.0])
    assert shares.iloc[1].isna().all()


def test_recovery_summary_identifies_known_step_without_monotonicity_assumption():
    rows = []
    for seed in range(4):
        rows.append({
            "scenario": "edge", "seed": seed, "D1": 0.9,
            "density": -0.1, "degree": 0.1, "block": 0.1, "edge": 0.8,
        })
        rows.append({
            "scenario": "block", "seed": seed, "D1": 0.6,
            "density": 0.0, "degree": 0.1, "block": 0.6, "edge": -0.1,
        })
    summary = recovery_summary(pd.DataFrame(rows)).set_index("scenario")
    assert summary.loc["edge", "dominant_mean_step"] == "edge"
    assert summary.loc["block", "dominant_mean_step"] == "block"
    assert bool(summary.loc["edge", "recovered"])
    assert summary.loc["edge", "mean_expected_share"] == pytest.approx(0.8)


def test_null_string_scenario_is_retained_as_negative_control():
    frame = pd.DataFrame([{
        "scenario": "null", "D1": 0.0,
        "density": 0.0, "degree": 0.0, "block": 0.0, "edge": 0.0,
    }])
    summary = recovery_summary(frame)
    assert summary.loc[0, "scenario"] == "null"
    assert summary.loc[0, "expected_step"] == "none"
    assert not bool(summary.loc[0, "recovered"])


def test_invalid_ladder_inputs_fail_loudly():
    with pytest.raises(ValueError, match="missing PCGA-L levels"):
        decompose_loss_levels(pd.DataFrame({"L0_none": [1.0]}))
    bad = _levels([[1.0, 0.9, np.nan, 0.8, 0.7]])
    with pytest.raises(ValueError, match="must all be finite"):
        decompose_loss_levels(bad)
