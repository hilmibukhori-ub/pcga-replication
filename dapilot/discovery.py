"""Stage 9 controlled evidence-family screening and statistical summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .ceiling import _fit_predict, _prediction_metrics
from .config import load_config
from .dgp import REGIME_NAMES, generate_world
from .diagnostics import _positions, _risk
from .evidence import (build_evidence_families, graph_counterfactual_predictions,
                       retrain_frozen_gcn)
from .experts import label_array
from .report import load_seeds
from .splits import TemporalSplits
from .stats import holm, paired_summary


FAMILIES = ("F1", "F2", "F3", "F4")


def _full_predictions(obs, pred_file):
    times = pred_file["decision_time_0based"]
    predictions = {}
    for name in pred_file.files:
        if name in ("decision_time_0based", "placebo_perms"):
            continue
        full = np.full(obs.returns.shape, np.nan, dtype=float)
        full[times] = pred_file[name]
        predictions[name] = full
    return times, predictions


def _subsets(regime):
    yield "all", np.ones(len(regime), dtype=bool)
    for value, name in enumerate(REGIME_NAMES):
        yield name, regime == value


def _screen_seed(source_run: Path, cfg, seed: int):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    world = generate_world(cfg, seed)
    obs, hidden = world.observable, world.hidden
    splits = TemporalSplits(cfg["splits"], obs.T)
    pred_file = np.load(source_run / f"seed_{seed:02d}" / "predictions.npz")
    feature_file = np.load(source_run / f"seed_{seed:02d}" / "selector_features.npz")
    times, predictions = _full_predictions(obs, pred_file)
    fit_pos = _positions(times, splits.decision_times("meta_train"))
    val_pos = _positions(times, splits.decision_times("selector_val"))
    cal_pos = _positions(times, splits.decision_times("calibration"))
    test_pos = _positions(times, splits.decision_times("test"))

    model, z_gcn, expert_info = retrain_frozen_gcn(obs, splits, cfg, seed)
    counterfactuals = graph_counterfactual_predictions(obs, model, z_gcn, seed)
    np.testing.assert_allclose(
        counterfactuals["base"][times], predictions["gcn"][times], rtol=1e-5, atol=1e-5,
        err_msg=f"seed {seed}: reconstructed GCN differs from frozen artifact",
    )
    families_full = build_evidence_families(obs, predictions, splits, counterfactuals)

    base_z = feature_file["Z"].astype(float)
    base_names = feature_file["names"].astype(str)
    base_cols = np.flatnonzero(~np.char.startswith(base_names, "S_"))
    decision_z = base_z[:, base_cols]
    mu = hidden.mu[times]
    p_g, p_n = predictions["gcn"][times], predictions["mlp"][times]
    oracle_benefit = ((mu - p_n) ** 2).mean(1) - ((mu - p_g) ** 2).mean(1)
    baseline_pred, baseline_hp = _fit_predict(
        "gbr", decision_z, oracle_benefit, fit_pos, val_pos, cal_pos, test_pos,
        np.arange(decision_z.shape[1]), cfg, seed,
    )
    baseline_weight = (baseline_pred > 0).astype(float)
    mu_test, p_g_test, p_n_test = mu[test_pos], p_g[test_pos], p_n[test_pos]
    baseline_step_risk = _risk(mu_test, p_g_test, p_n_test, baseline_weight)
    regime_test = hidden.regime[times[test_pos]]
    truth_test = oracle_benefit[test_pos]

    rows = []
    for family in FAMILIES:
        values, names = families_full[family]
        family_z = values[times]
        if not np.isfinite(family_z[np.r_[fit_pos, val_pos, cal_pos, test_pos]]).all():
            raise ValueError(f"seed {seed} {family}: unavailable feature rows")
        augmented = np.concatenate([decision_z, family_z], axis=1)
        candidate_pred, hp = _fit_predict(
            "gbr", augmented, oracle_benefit, fit_pos, val_pos, cal_pos, test_pos,
            np.arange(augmented.shape[1]), cfg, seed,
        )
        candidate_weight = (candidate_pred > 0).astype(float)
        candidate_step_risk = _risk(mu_test, p_g_test, p_n_test, candidate_weight)
        baseline_metrics = _prediction_metrics(baseline_pred, truth_test)
        candidate_metrics = _prediction_metrics(candidate_pred, truth_test)
        for subset, mask in _subsets(regime_test):
            rows.append({
                "seed": seed, "family": family, "subset": subset,
                "n_steps": int(mask.sum()),
                "baseline_risk": float(baseline_step_risk[mask].mean()),
                "candidate_risk": float(candidate_step_risk[mask].mean()),
                "risk_diff": float((candidate_step_risk[mask] - baseline_step_risk[mask]).mean()),
                "baseline_corr": baseline_metrics["corr"],
                "candidate_corr": candidate_metrics["corr"],
                "corr_diff": candidate_metrics["corr"] - baseline_metrics["corr"],
                "baseline_r2": baseline_metrics["r2"],
                "candidate_r2": candidate_metrics["r2"],
                "feature_count": len(names),
                "baseline_hyperparams": baseline_hp,
                "candidate_hyperparams": hp,
                "reconstructed_gcn": expert_info,
            })
    return rows


def _screen_summary(rows, alpha=0.05):
    summaries = []
    for family in FAMILIES:
        for subset in ("all",) + REGIME_NAMES:
            chosen = sorted(
                (r for r in rows if r["family"] == family and r["subset"] == subset),
                key=lambda r: r["seed"],
            )
            for metric in ("risk_diff", "corr_diff"):
                summary = paired_summary(np.array([r[metric] for r in chosen]))
                summaries.append({"family": family, "subset": subset, "metric": metric,
                                  **summary})
    primary = [r for r in summaries if r["subset"] == "all" and r["metric"] == "risk_diff"]
    adjusted = holm([r["wilcoxon_p"] for r in primary])
    for row, p_adj in zip(primary, adjusted):
        row["p_holm"] = p_adj
        if row["mean"] < 0 and row["t_ci_high"] < 0 and p_adj < alpha:
            row["screening"] = "pass"
        elif row["mean"] < 0:
            row["screening"] = "inconclusive"
        else:
            row["screening"] = "not_supported"
    return summaries


def run_oracle_screen(source_run: str | Path, output_dir: str | Path) -> dict:
    source_run, output_dir = Path(source_run), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(source_run / "config.yaml")
    seed_metrics = load_seeds(source_run, cfg)
    rows = []
    for metric in seed_metrics:
        rows.extend(_screen_seed(source_run, cfg, metric["seed"]))
    summaries = _screen_summary(rows, cfg["evaluation"]["alpha"])
    primary = [r for r in summaries if r["subset"] == "all" and r["metric"] == "risk_diff"]
    passed = [r["family"] for r in primary if r["screening"] == "pass"]
    result = {
        "source_run": source_run.name,
        "status": "post_hoc_oracle_supervised_discovery_screen",
        "primary_rule": "hard graph-vs-mlp selection from oracle-B* supervised GBR",
        "multiple_testing": "Holm across F1-F4 primary all-regime risk comparisons",
        "passed_families": passed,
        "summary": summaries,
        "per_seed": rows,
    }
    with open(output_dir / "oracle_screen.json", "w") as f:
        json.dump(result, f, indent=1)
    flat_keys = [k for k, v in rows[0].items() if not isinstance(v, dict)]
    with open(output_dir / "oracle_screen_per_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in flat_keys} for r in rows])
    with open(output_dir / "oracle_screen_ablation.csv", "w", newline="") as f:
        summary_fields = list(dict.fromkeys(k for row in summaries for k in row))
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    if not passed:
        independent = {
            "status": "not_run_no_family_passed_oracle_screen",
            "passed_families": [],
            "holdout_opened": False,
            "reason": ("Independent realised-supervision evaluation is gated on a family passing "
                       "the Holm-corrected oracle decision-risk screen."),
        }
        with open(output_dir / "independent_evaluation_summary.json", "w") as f:
            json.dump(independent, f, indent=1)
    return result
