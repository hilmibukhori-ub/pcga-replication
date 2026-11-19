"""Post-pilot diagnostics that reuse frozen v2 artefacts without changing pilot verdicts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge

from .config import load_config
from .dgp import generate_world
from .experts import label_array
from .report import load_seeds
from .selectors import fit_affine
from .splits import TemporalSplits
from .stats import paired_summary


TEMPERATURE_MULTIPLIERS = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])


def _positions(times: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(times, wanted)
    if np.any(pos >= len(times)) or not np.array_equal(times[pos], wanted):
        raise ValueError("requested split is not present in saved decision times")
    return pos


def _risk(mu, p_g, p_n, weight):
    pred = weight[:, None] * p_g + (1 - weight[:, None]) * p_n
    return ((mu - pred) ** 2).mean(1)


def tune_soft_temperature(benefit, p_g, p_n, y, cal_pos):
    """Tune only the hard-to-soft mapping; the benefit forecast stays completely fixed."""
    scale = max(float(np.std(benefit[cal_pos])), 1e-8)
    candidates = scale * TEMPERATURE_MULTIPLIERS
    losses = []
    for temperature in candidates:
        w = expit(benefit[cal_pos] / temperature)
        pred = w[:, None] * p_g[cal_pos] + (1 - w[:, None]) * p_n[cal_pos]
        losses.append(float(((y[cal_pos] - pred) ** 2).mean()))
    best = int(np.argmin(losses))
    return float(candidates[best]), losses


def _ridge_benefit(z, target, fit_pos, val_pos, cal_pos, test_pos, cols, alphas, calibration):
    x = z[:, cols]
    mean, std = x[fit_pos].mean(0), x[fit_pos].std(0)
    std[std < 1e-10] = 1.0
    x = (x - mean) / std
    bm, bs = target[fit_pos].mean(), target[fit_pos].std()
    scaled = (target - bm) / bs
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(x[fit_pos], scaled[fit_pos])
        loss = float(((model.predict(x[val_pos]) - scaled[val_pos]) ** 2).mean())
        if best is None or loss < best[0]:
            best = loss, alpha, model
    raw_cal = best[2].predict(x[cal_pos]) * bs + bm
    raw_test = best[2].predict(x[test_pos]) * bs + bm
    a, c, mode = fit_affine(raw_cal, target[cal_pos], calibration)
    pred_test = a + c * raw_test
    return pred_test, {"alpha": best[1], "val_mse_std": best[0],
                       "calib_a": a, "calib_c": c, "calib_mode": mode}


def run_diagnostics(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    metrics = load_seeds(run_dir, cfg)  # exact seed/hash guard
    rows = []

    for metric in metrics:
        seed = metric["seed"]
        world = generate_world(cfg, seed)
        obs, hidden = world.observable, world.hidden
        splits = TemporalSplits(cfg["splits"], obs.T)
        pred_file = np.load(run_dir / f"seed_{seed:02d}" / "predictions.npz")
        selector_file = np.load(run_dir / f"seed_{seed:02d}" / "selector_outputs.npz")
        feature_file = np.load(run_dir / f"seed_{seed:02d}" / "selector_features.npz")
        times = pred_file["decision_time_0based"]
        if not np.array_equal(times, selector_file["decision_time_0based"]):
            raise ValueError(f"seed {seed}: prediction/selector timelines differ")

        fit_pos = _positions(times, splits.decision_times("meta_train"))
        val_pos = _positions(times, splits.decision_times("selector_val"))
        cal_pos = _positions(times, splits.decision_times("calibration"))
        test_pos = _positions(times, splits.decision_times("test"))
        p_g, p_n = pred_file["gcn"].astype(float), pred_file["mlp"].astype(float)
        y_full = label_array(obs, splits.horizon).astype(float)
        y = y_full[times]
        mu_test = hidden.mu[times[test_pos]]

        # Hard versus soft with an identical, already-frozen A2a benefit forecast.
        benefit = selector_file["real/A2a/benefit"].astype(float)
        temperature, temperature_losses = tune_soft_temperature(
            benefit, p_g, p_n, y, cal_pos
        )
        hard_w = (benefit[test_pos] > 0).astype(float)
        soft_w = expit(benefit[test_pos] / temperature)
        hard_risk = float(_risk(mu_test, p_g[test_pos], p_n[test_pos], hard_w).mean())
        soft_risk = float(_risk(mu_test, p_g[test_pos], p_n[test_pos], soft_w).mean())

        # Incremental structural information with the exact same deterministic estimator.
        z = feature_file["Z"].astype(float)
        names = feature_file["names"].astype(str)
        if not np.isfinite(z[np.r_[fit_pos, val_pos, cal_pos, test_pos]]).all():
            raise ValueError(f"seed {seed}: non-finite diagnostic features")
        all_cols = np.arange(z.shape[1])
        nonstruct_cols = np.flatnonzero(~np.char.startswith(names, "S_"))
        realised_benefit = (((y - p_n) ** 2).mean(1)
                            - ((y - p_g) ** 2).mean(1))
        oracle_benefit = (_risk(mu_test, p_g[test_pos], p_n[test_pos], np.zeros(len(test_pos)))
                          - _risk(mu_test, p_g[test_pos], p_n[test_pos], np.ones(len(test_pos))))
        ridge_results = {}
        for label, cols in (("nonstruct", nonstruct_cols), ("all", all_cols)):
            predicted, hp = _ridge_benefit(
                z, realised_benefit, fit_pos, val_pos, cal_pos, test_pos, cols,
                cfg["selectors"]["ridge_alpha_grid"], cfg["selectors"]["calibration"],
            )
            w = (predicted > 0).astype(float)
            ridge_results[label] = {
                "risk": float(_risk(mu_test, p_g[test_pos], p_n[test_pos], w).mean()),
                "benefit_rmse": float(np.sqrt(((predicted - oracle_benefit) ** 2).mean())),
                "benefit_corr": float(np.corrcoef(predicted, oracle_benefit)[0, 1]),
                "hyperparams": hp,
            }

        rows.append({
            "seed": seed,
            "hard_risk": hard_risk,
            "soft_risk": soft_risk,
            "soft_minus_hard_risk": soft_risk - hard_risk,
            "temperature": temperature,
            "temperature_calibration_losses": temperature_losses,
            "ridge_nonstruct_risk": ridge_results["nonstruct"]["risk"],
            "ridge_all_risk": ridge_results["all"]["risk"],
            "ridge_all_minus_nonstruct_risk": (ridge_results["all"]["risk"]
                                                - ridge_results["nonstruct"]["risk"]),
            "ridge_nonstruct_benefit_rmse": ridge_results["nonstruct"]["benefit_rmse"],
            "ridge_all_benefit_rmse": ridge_results["all"]["benefit_rmse"],
            "ridge_all_minus_nonstruct_rmse": (ridge_results["all"]["benefit_rmse"]
                                                - ridge_results["nonstruct"]["benefit_rmse"]),
            "ridge_nonstruct_benefit_corr": ridge_results["nonstruct"]["benefit_corr"],
            "ridge_all_benefit_corr": ridge_results["all"]["benefit_corr"],
            "ridge_hyperparams": {k: v["hyperparams"] for k, v in ridge_results.items()},
        })

    summaries = {}
    for key in ("soft_minus_hard_risk", "ridge_all_minus_nonstruct_risk",
                "ridge_all_minus_nonstruct_rmse"):
        summaries[key] = paired_summary(np.array([r[key] for r in rows]))
    result = {
        "run": run_dir.name,
        "status": "post_hoc_diagnostic_not_primary_evidence",
        "temperature_multipliers": TEMPERATURE_MULTIPLIERS.tolist(),
        "summaries": summaries,
        "per_seed": rows,
    }
    with open(run_dir / "diagnostics.json", "w") as f:
        json.dump(result, f, indent=1)
    flat_keys = [k for k, v in rows[0].items() if not isinstance(v, (dict, list))]
    with open(run_dir / "diagnostics_per_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in flat_keys} for r in rows])
    return result
