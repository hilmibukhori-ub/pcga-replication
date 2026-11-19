"""Locked-test evaluation. This module is the only place hidden quantities (mu, regime) are
used, and it is only called after every selector has been fitted and frozen."""

from __future__ import annotations

import numpy as np

from .dgp import REGIME_NAMES


def cond_risk(mu: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-step conditional risk (excess over irreducible noise): mean_i (mu - pred)^2."""
    return ((mu - pred) ** 2).mean(1)


def realised_mse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((y - pred) ** 2).mean(1)


def _subsets(regime: np.ndarray):
    yield "all", np.ones_like(regime, dtype=bool)
    for r, n in enumerate(REGIME_NAMES):
        yield n, regime == r


def calibration_stats(pred: np.ndarray, target: np.ndarray, bins: int) -> dict:
    if len(pred) < 3 or np.std(pred) < 1e-12:
        return {"slope": np.nan, "intercept": np.nan, "ece": np.nan}
    slope, intercept = np.polyfit(pred, target, 1)
    edges = np.quantile(pred, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(pred[m].mean() - target[m].mean())
    return {"slope": float(slope), "intercept": float(intercept), "ece": float(ece)}


def evaluate_selectors(outputs: dict, pG, pN, y, mu, regime, test_idx, bins=10, RP=None) -> dict:
    """outputs: name -> {"w": (T,), "benefit": (T,) | None, "d2": (T,) | None}.
    RP: (T,) mean conditional risk of the placebo experts (for D2 estimation error).
    Returns name -> subset -> metric -> value, plus the oracle."""
    RPt = RP[test_idx] if RP is not None else None
    pG, pN, y, mu, reg = pG[test_idx], pN[test_idx], y[test_idx], mu[test_idx], regime[test_idx]
    RG, RN = cond_risk(mu, pG), cond_risk(mu, pN)
    R_or = np.minimum(RG, RN)
    b_star = RN - RG
    g_star = (RG < RN).astype(float)
    res = {}
    all_out = dict(outputs)
    all_out["Oracle"] = {"w": None, "benefit": None, "d2": None}
    for name, o in all_out.items():
        w = g_star if name == "Oracle" else o["w"][test_idx]
        if not np.isfinite(w).all():
            raise ValueError(f"{name}: undefined weights in locked test")
        pred = w[:, None] * pG + (1 - w[:, None]) * pN
        R = cond_risk(mu, pred)
        L = realised_mse(y, pred)
        hard = np.isin(w, (0.0, 1.0)).all()
        per = {}
        for sub, m in _subsets(reg):
            if not m.any():
                continue
            d = {
                "n_steps": int(m.sum()),
                "regret": float((R[m] - R_or[m]).mean()),
                "cond_risk": float(R[m].mean()),
                "realised_mse": float(L[m].mean()),
                "graph_weight": float(w[m].mean()),
                "oracle_graph_freq": float(g_star[m].mean()),
                "oracle_agreement": float(((w[m] > 0.5) == (g_star[m] > 0.5)).mean()) if hard else np.nan,
            }
            if o["benefit"] is not None:
                bh = o["benefit"][test_idx][m]
                err = bh - b_star[m]
                d["benefit_rmse"] = float(np.sqrt((err**2).mean()))
                d["benefit_bias"] = float(err.mean())
                d["benefit_corr"] = float(np.corrcoef(bh, b_star[m])[0, 1]) if bh.std() > 0 else np.nan
                cal = calibration_stats(bh, b_star[m], bins)
                d.update({f"calib_{k}": v for k, v in cal.items()})
            if o["d2"] is not None and RPt is not None:
                d2_star = RPt[m] - RG[m]
                d["d2_rmse"] = float(np.sqrt(((o["d2"][test_idx][m] - d2_star) ** 2).mean()))
            per[sub] = d
        res[name] = per
    return res


def evaluate_graph_value(preds: dict, y, mu, regime, test_idx, graph="gcn",
                         nongraph=("mlp", "ridge", "cross_mlp"), placebos=()) -> dict:
    """D1 / D2, realised (label-based) and oracle (conditional-risk based), plus expert table."""
    y, mu, reg = y[test_idx], mu[test_idx], regime[test_idx]
    L = {k: realised_mse(y, p[test_idx]) for k, p in preds.items()}
    R = {k: cond_risk(mu, p[test_idx]) for k, p in preds.items()}
    LP = np.mean([L[k] for k in placebos], 0)
    RP = np.mean([R[k] for k in placebos], 0)
    out = {}
    for sub, m in _subsets(reg):
        if not m.any():
            continue
        d = {}
        for ng in nongraph:
            d[f"D1_realised_vs_{ng}"] = float((L[ng][m] - L[graph][m]).mean())
            d[f"D1_oracle_vs_{ng}"] = float((R[ng][m] - R[graph][m]).mean())
            d[f"D1placebo_realised_vs_{ng}"] = float((L[ng][m] - LP[m]).mean())
            d[f"D1placebo_oracle_vs_{ng}"] = float((R[ng][m] - RP[m]).mean())
        d["D2_realised"] = float((LP[m] - L[graph][m]).mean())
        d["D2_oracle"] = float((RP[m] - R[graph][m]).mean())
        for k in preds:
            d[f"expert_mse_{k}"] = float(L[k][m].mean())
            d[f"expert_risk_{k}"] = float(R[k][m].mean())
        out[sub] = d
    return out
