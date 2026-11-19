"""Core estimands and recovery diagnostics for hierarchical PCGA (PCGA-L).

PCGA-L is a performance-attribution procedure, not a forecasting architecture.  It
compares matched, fully retrained experts along an ordered ladder of graph controls:

    L0 no graph -> L1 density -> L2 degree -> L3 block -> L4 observed graph.

For loss metrics (smaller is better), a step is the loss at the less constrained
level minus the loss at the more constrained level.  Positive values therefore mean
that the newly admitted structure helped.  Steps may be negative; monotonicity is not
an assumption.  The four steps telescope exactly to total graph utility (L0 - L4).

This module deliberately contains no significance-test shortcut.  An exact placebo
claim requires the experiment-specific exchangeability argument and retrained placebo
distribution; it cannot be inferred from a table of mean losses alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


LEVELS = ("L0_none", "L1_er", "L2_degree", "L3_block", "L4_real")
STEP_PAIRS = {
    "density": ("L0_none", "L1_er"),
    "degree": ("L1_er", "L2_degree"),
    "block": ("L2_degree", "L3_block"),
    "edge": ("L3_block", "L4_real"),
}
STEPS = tuple(STEP_PAIRS)
DEFAULT_EXPECTED_STEP = {"edge": "edge", "block": "block", "degree": "degree"}


def _require_finite_levels(df: pd.DataFrame, levels: Sequence[str]) -> None:
    missing = [name for name in levels if name not in df]
    if missing:
        raise ValueError(f"missing PCGA-L levels: {missing}")
    values = df[list(levels)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("PCGA-L level losses must all be finite")


def decompose_loss_levels(
    df: pd.DataFrame,
    *,
    levels: Sequence[str] = LEVELS,
) -> pd.DataFrame:
    """Add PCGA-L step effects and total graph utility to a loss table.

    Parameters
    ----------
    df:
        One row per independent run, with ordered L0--L4 loss columns.
    levels:
        The five ordered loss columns.  The default is the repository convention.

    Returns
    -------
    A copy with ``density``, ``degree``, ``block``, ``edge`` and ``D1``.
    Positive effects mean lower loss after admitting the corresponding structure.
    """
    if len(levels) != 5:
        raise ValueError("PCGA-L requires exactly five ordered levels (L0 through L4)")
    _require_finite_levels(df, levels)
    out = df.copy()
    for step, (left, right) in zip(STEPS, zip(levels[:-1], levels[1:])):
        out[step] = out[left].astype(float) - out[right].astype(float)
    out["D1"] = out[levels[0]].astype(float) - out[levels[-1]].astype(float)
    return out


def telescoping_error(df: pd.DataFrame) -> np.ndarray:
    """Return D1 minus the sum of the four PCGA-L steps for every row."""
    missing = [name for name in ("D1", *STEPS) if name not in df]
    if missing:
        raise ValueError(f"missing decomposed PCGA-L columns: {missing}")
    values = df[["D1", *STEPS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("PCGA-L decomposition must be finite")
    return df["D1"].to_numpy(dtype=float) - df[list(STEPS)].sum(axis=1).to_numpy(dtype=float)


def positive_step_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Shares of positive attributed value, leaving all-zero rows undefined.

    Negative steps are retained in the decomposition but contribute zero to this
    descriptive concentration measure.  Shares are not statistical evidence.
    """
    missing = [name for name in STEPS if name not in df]
    if missing:
        raise ValueError(f"missing PCGA-L steps: {missing}")
    positive = df[list(STEPS)].astype(float).clip(lower=0)
    denominator = positive.sum(axis=1).replace(0.0, np.nan)
    return positive.div(denominator, axis=0)


def recovery_summary(
    df: pd.DataFrame,
    *,
    scenario_col: str = "scenario",
    expected_steps: Mapping[str, str] = DEFAULT_EXPECTED_STEP,
    share_threshold: float = 0.70,
) -> pd.DataFrame:
    """Summarise whether controlled scenarios recover their known structural level.

    This is a construct-recovery diagnostic.  It is intentionally separate from the
    empirical exact placebo test and should not be described as market evidence.
    """
    if not 0 < share_threshold <= 1:
        raise ValueError("share_threshold must lie in (0, 1]")
    if scenario_col not in df:
        raise ValueError(f"missing scenario column: {scenario_col}")
    bad = {scenario: step for scenario, step in expected_steps.items() if step not in STEPS}
    if bad:
        raise ValueError(f"unknown expected PCGA-L steps: {bad}")

    shares = positive_step_shares(df)
    rows = []
    for scenario, group in df.groupby(scenario_col, sort=False):
        idx = group.index
        mean_steps = group[list(STEPS)].mean()
        dominant = mean_steps.idxmax()
        row = {
            "scenario": scenario,
            "n": int(len(group)),
            "mean_D1": float(group["D1"].mean()),
            **{f"mean_{step}": float(mean_steps[step]) for step in STEPS},
            "dominant_mean_step": dominant,
        }
        expected = expected_steps.get(str(scenario))
        row["expected_step"] = expected if expected is not None else "none"
        if expected is None:
            row["mean_expected_share"] = np.nan
            row["fraction_share_at_least_threshold"] = np.nan
            row["recovered"] = False
        else:
            expected_share = shares.loc[idx, expected]
            row["mean_expected_share"] = float(expected_share.mean())
            row["fraction_share_at_least_threshold"] = float(
                (expected_share >= share_threshold).mean()
            )
            row["recovered"] = bool(dominant == expected)
        rows.append(row)
    return pd.DataFrame(rows)

