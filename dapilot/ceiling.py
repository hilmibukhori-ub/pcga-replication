"""Predictability-ceiling analysis for conditional graph benefit.

This is a synthetic, post-hoc diagnostic. Hidden conditional means may be used as training labels
only to estimate an empirical information ceiling; they never enter the primary pilot models.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from .config import load_config
from .dgp import generate_world
from .diagnostics import _positions, _risk, tune_soft_temperature
from .experts import label_array
from .report import load_seeds
from .selectors import fit_affine
from .splits import TemporalSplits
from .stats import paired_summary


def optimal_soft_weight(mu: np.ndarray, p_g: np.ndarray, p_n: np.ndarray) -> np.ndarray:
    """Per-time convex weight minimizing conditional risk of w*G + (1-w)*N."""
    direction = p_g - p_n
    numerator = ((mu - p_n) * direction).sum(1)
    denominator = (direction**2).sum(1)
    return np.clip(numerator / np.maximum(denominator, 1e-12), 0.0, 1.0)


def feature_groups(names: np.ndarray) -> dict[str, np.ndarray]:
    names = names.astype(str)
    structural = np.flatnonzero(np.char.startswith(names, "S_"))
    history = np.flatnonzero(np.char.startswith(names, "E_"))
    disagreement = np.flatnonzero(np.char.startswith(names, "Q_"))
    market = np.flatnonzero(np.char.startswith(names, "M_"))
    decision = np.sort(np.r_[history, disagreement, market])
    return {
        "structural": structural,
        "history": history,
        "decision": decision,
        "all": np.arange(len(names)),
    }


def _standardise_features(z, fit_pos, cols):
    x = z[:, cols]
    mean, std = x[fit_pos].mean(0), x[fit_pos].std(0)
    std[std < 1e-10] = 1.0
    return (x - mean) / std


def _fit_predict(model_kind, z, target, fit_pos, val_pos, cal_pos, test_pos, cols, cfg, seed):
    x = _standardise_features(z, fit_pos, cols)
    mean, std = target[fit_pos].mean(), target[fit_pos].std()
    std = max(float(std), 1e-12)
    scaled = (target - mean) / std
    candidates = []
    if model_kind == "ridge":
        for alpha in cfg["selectors"]["ridge_alpha_grid"]:
            model = Ridge(alpha=alpha).fit(x[fit_pos], scaled[fit_pos])
            loss = float(((model.predict(x[val_pos]) - scaled[val_pos]) ** 2).mean())
            candidates.append((loss, {"alpha": alpha}, model))
    elif model_kind == "gbr":
        for depth in (1, 2, 3):
            for min_leaf in (10, 30):
                model = GradientBoostingRegressor(
                    learning_rate=0.05, n_estimators=150, max_depth=depth,
                    min_samples_leaf=min_leaf, loss="squared_error", random_state=seed,
                ).fit(x[fit_pos], scaled[fit_pos])
                loss = float(((model.predict(x[val_pos]) - scaled[val_pos]) ** 2).mean())
                candidates.append((loss, {"max_depth": depth, "min_samples_leaf": min_leaf}, model))
    else:
        raise ValueError(model_kind)
    val_loss, hp, model = min(candidates, key=lambda item: item[0])
    raw_cal = model.predict(x[cal_pos]) * std + mean
    raw_test = model.predict(x[test_pos]) * std + mean
    a, c, mode = fit_affine(raw_cal, target[cal_pos], cfg["selectors"]["calibration"])
    return a + c * raw_test, {**hp, "val_mse_std": val_loss,
                              "calib_a": a, "calib_c": c, "calib_mode": mode}


def _prediction_metrics(predicted, truth):
    error = predicted - truth
    variance = ((truth - truth.mean()) ** 2).sum()
    return {
        "rmse": float(np.sqrt((error**2).mean())),
        "corr": float(np.corrcoef(predicted, truth)[0, 1]) if predicted.std() > 0 else np.nan,
        "r2": float(1 - (error**2).sum() / variance) if variance > 0 else np.nan,
    }


def run_ceiling_analysis(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_metrics = load_seeds(run_dir, cfg)
    prediction_rows, decision_rows = [], []

    for seed_metric in seed_metrics:
        seed = seed_metric["seed"]
        world = generate_world(cfg, seed)
        obs, hidden = world.observable, world.hidden
        splits = TemporalSplits(cfg["splits"], obs.T)
        pred_file = np.load(run_dir / f"seed_{seed:02d}" / "predictions.npz")
        selector_file = np.load(run_dir / f"seed_{seed:02d}" / "selector_outputs.npz")
        feature_file = np.load(run_dir / f"seed_{seed:02d}" / "selector_features.npz")
        times = pred_file["decision_time_0based"]
        fit_pos = _positions(times, splits.decision_times("meta_train"))
        val_pos = _positions(times, splits.decision_times("selector_val"))
        cal_pos = _positions(times, splits.decision_times("calibration"))
        test_pos = _positions(times, splits.decision_times("test"))
        p_g, p_n = pred_file["gcn"].astype(float), pred_file["mlp"].astype(float)
        mu = hidden.mu[times]
        y = label_array(obs, splits.horizon).astype(float)[times]
        r_g = ((mu - p_g) ** 2).mean(1)
        r_n = ((mu - p_n) ** 2).mean(1)
        oracle_benefit = r_n - r_g
        realised_benefit = (((y - p_n) ** 2).mean(1) - ((y - p_g) ** 2).mean(1))
        z = feature_file["Z"].astype(float)
        groups = feature_groups(feature_file["names"])

        fitted = {}
        for supervision, target in (("oracle", oracle_benefit), ("realised", realised_benefit)):
            for model in ("ridge", "gbr"):
                for group, cols in groups.items():
                    predicted, hp = _fit_predict(
                        model, z, target, fit_pos, val_pos, cal_pos, test_pos, cols, cfg, seed
                    )
                    truth = oracle_benefit[test_pos]
                    metrics = _prediction_metrics(predicted, truth)
                    hard_w = (predicted > 0).astype(float)
                    hard_risk = float(_risk(
                        mu[test_pos], p_g[test_pos], p_n[test_pos], hard_w
                    ).mean())
                    fitted[(supervision, model, group)] = (predicted, hard_risk)
                    prediction_rows.append({
                        "seed": seed, "supervision": supervision, "model": model,
                        "features": group, **metrics, "hard_risk": hard_risk,
                        "hyperparams": hp,
                    })

        mu_test, p_g_test, p_n_test = mu[test_pos], p_g[test_pos], p_n[test_pos]
        exact_hard = (oracle_benefit[test_pos] > 0).astype(float)
        exact_soft = optimal_soft_weight(mu_test, p_g_test, p_n_test)
        a2a_benefit = selector_file["real/A2a/benefit"].astype(float)
        a2a_hard = selector_file["real/A2a/w"].astype(float)[test_pos]
        temperature, _ = tune_soft_temperature(a2a_benefit, p_g, p_n, y, cal_pos)
        a2a_soft = expit(a2a_benefit[test_pos] / temperature)
        gate = selector_file["real/Gate/w"].astype(float)[test_pos]
        fixed = selector_file["real/FixedEnsemble/w"].astype(float)[test_pos]
        oracle_feature_risk = fitted[("oracle", "gbr", "all")][1]
        realised_feature_risk = fitted[("realised", "gbr", "all")][1]
        risks = {
            "always_n": float(r_n[test_pos].mean()),
            "always_g": float(r_g[test_pos].mean()),
            "fixed_ensemble": float(_risk(mu_test, p_g_test, p_n_test, fixed).mean()),
            "gate": float(_risk(mu_test, p_g_test, p_n_test, gate).mean()),
            "a2a_hard": float(_risk(mu_test, p_g_test, p_n_test, a2a_hard).mean()),
            "a2a_soft_same_forecast": float(_risk(mu_test, p_g_test, p_n_test, a2a_soft).mean()),
            "feature_oracle_gbr_hard": oracle_feature_risk,
            "feature_realised_gbr_hard": realised_feature_risk,
            "exact_benefit_hard": float(_risk(mu_test, p_g_test, p_n_test, exact_hard).mean()),
            "exact_optimal_soft": float(_risk(mu_test, p_g_test, p_n_test, exact_soft).mean()),
        }
        decision_rows.append({
            "seed": seed, **risks,
            "empirical_information_gap": oracle_feature_risk - risks["exact_benefit_hard"],
            "noisy_label_estimation_gap": realised_feature_risk - oracle_feature_risk,
            "current_estimator_gap": risks["a2a_hard"] - realised_feature_risk,
            "hard_to_soft_gap": risks["a2a_soft_same_forecast"] - risks["a2a_hard"],
            "soft_oracle_advantage": risks["exact_optimal_soft"] - risks["exact_benefit_hard"],
        })

    prediction_summary = []
    scalar_prediction_keys = ("rmse", "corr", "r2", "hard_risk")
    combinations = sorted({(r["supervision"], r["model"], r["features"])
                           for r in prediction_rows})
    for supervision, model, features in combinations:
        chosen = [r for r in prediction_rows if (r["supervision"], r["model"], r["features"])
                  == (supervision, model, features)]
        for metric in scalar_prediction_keys:
            summary = paired_summary(np.array([r[metric] for r in chosen]))
            prediction_summary.append({
                "supervision": supervision, "model": model, "features": features,
                "metric": metric, **summary,
            })

    decision_summary = {}
    decision_keys = [k for k, v in decision_rows[0].items()
                     if k != "seed" and isinstance(v, (float, int))]
    for key in decision_keys:
        decision_summary[key] = paired_summary(np.array([r[key] for r in decision_rows]))

    result = {
        "run": run_dir.name,
        "status": "post_hoc_synthetic_predictability_ceiling_not_primary_evidence",
        "prediction_summary": prediction_summary,
        "decision_summary": decision_summary,
        "prediction_per_seed": prediction_rows,
        "decision_per_seed": decision_rows,
    }
    with open(run_dir / "predictability_ceiling.json", "w") as f:
        json.dump(result, f, indent=1)

    flat_prediction = [{k: v for k, v in r.items() if k != "hyperparams"}
                       for r in prediction_rows]
    with open(run_dir / "predictability_ceiling_per_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_prediction[0].keys())
        writer.writeheader()
        writer.writerows(flat_prediction)
    return result
