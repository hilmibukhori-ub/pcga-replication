"""Causal decision-relevant evidence families for Stage 9.

Every label-based row at decision time t uses only predictions s whose labels have matured by t.
Graph sensitivity uses only the current observed adjacency, its lag, and deterministic perturbation
noise; hidden graph quantities are never accepted by this module.
"""

from __future__ import annotations

import numpy as np
import torch

from .dgp import ObservableData
from .experts import (_predict, build_inputs, label_array, standardise,
                      train_graphnet)
from .graphs import mean_aggregation, symmetric_from_upper
from .splits import TemporalSplits


def _stack(names, columns):
    return np.stack(columns, 1).astype(np.float64), names


def _matured_bounds(t: int, window: int, horizon: int, valid_from: int):
    hi = t - horizon
    lo = hi - window + 1
    if lo < valid_from:
        return None
    return slice(lo, hi + 1)


def local_residual_reliability(obs, preds, splits, windows=(5, 20, 60)):
    """F1: cross-sectional distribution of per-asset rolling loss and benefit."""
    y = label_array(obs, splits.horizon).astype(float)
    e_g2 = (y - preds["gcn"]) ** 2
    e_n2 = (y - preds["mlp"]) ** 2
    valid_from = splits.start("expert_val")
    names, columns = [], []
    for window in windows:
        stats = np.full((obs.T, 7), np.nan)
        for t in range(obs.T):
            bounds = _matured_bounds(t, window, splits.horizon, valid_from)
            if bounds is None:
                continue
            local_g = e_g2[bounds].mean(0)
            local_n = e_n2[bounds].mean(0)
            benefit = local_n - local_g
            stats[t] = [
                np.quantile(benefit, 0.10), np.quantile(benefit, 0.25),
                np.median(benefit), np.quantile(benefit, 0.75),
                np.quantile(benefit, 0.90), benefit.std(), (benefit > 0).mean(),
            ]
        labels = ("benefit_q10", "benefit_q25", "benefit_median", "benefit_q75",
                  "benefit_q90", "benefit_xs_std", "benefit_frac_positive")
        for j, label in enumerate(labels):
            names.append(f"F1_{label}_w{window}")
            columns.append(stats[:, j])
    return _stack(names, columns)


def conditional_error_covariance(obs, preds, splits, windows=(20, 60), ridge=1e-8):
    """F2: chronological error moments and stabilized candidate mixture weights."""
    y = label_array(obs, splits.horizon).astype(float)
    e_g, e_n = y - preds["gcn"], y - preds["mlp"]
    valid_from = splits.start("expert_val")
    names, columns = [], []
    for window in windows:
        stats = np.full((obs.T, 6), np.nan)
        for t in range(obs.T):
            bounds = _matured_bounds(t, window, splits.horizon, valid_from)
            if bounds is None:
                continue
            g, n = e_g[bounds].ravel(), e_n[bounds].ravel()
            vg, vn, cov = np.mean(g * g), np.mean(n * n), np.mean(g * n)
            scale = max(vg + vn, ridge)
            stabilized = max(vg + vn - 2 * cov, ridge * scale)
            weight = np.clip((vn - cov) / stabilized, 0.0, 1.0)
            corr = cov / np.sqrt(max(vg * vn, ridge * scale * scale))
            stats[t] = [cov / scale, corr, stabilized / scale, weight,
                        (vg - cov) / scale, (vn - cov) / scale]
        labels = ("cov_scaled", "error_corr", "diversification_scaled", "mix_weight",
                  "g_unique_variance", "n_unique_variance")
        for j, label in enumerate(labels):
            names.append(f"F2_{label}_w{window}")
            columns.append(stats[:, j])
    return _stack(names, columns)


def disagreement_distribution(obs, preds, splits, windows=(20, 60)):
    """F4: signed/current and rolling disagreement distribution beyond MAD/RMS."""
    signed = (preds["gcn"] - preds["mlp"]).astype(float)
    magnitude = np.abs(signed)
    valid_from = splits.start("expert_val")
    names, columns = [], []

    current = np.stack([
        np.quantile(signed, 0.10, axis=1), np.median(signed, axis=1),
        np.quantile(signed, 0.90, axis=1), signed.std(1),
        (signed > 0).mean(1), np.quantile(signed, 0.75, axis=1) - np.quantile(signed, 0.25, axis=1),
    ], 1)
    for j, label in enumerate(("signed_q10", "signed_median", "signed_q90", "signed_xs_std",
                               "signed_frac_positive", "signed_iqr")):
        names.append(f"F4_current_{label}")
        columns.append(current[:, j])

    for window in windows:
        stats = np.full((obs.T, 5), np.nan)
        for t in range(obs.T):
            lo = t - window + 1
            if lo < valid_from:
                continue
            s = signed[lo:t + 1].ravel()
            a = magnitude[lo:t + 1].ravel()
            stats[t] = [np.quantile(s, 0.10), np.median(s), np.quantile(s, 0.90),
                        np.median(a), np.quantile(a, 0.90)]
        for j, label in enumerate(("signed_q10", "signed_median", "signed_q90",
                                   "magnitude_median", "magnitude_q90")):
            names.append(f"F4_rolling_{label}_w{window}")
            columns.append(stats[:, j])
    return _stack(names, columns)


def _drop_edges(adjacency: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Deterministically drop current observed edges; never add or inspect hidden edges."""
    T, n, _ = adjacency.shape
    iu = np.triu_indices(n, 1)
    out = np.zeros_like(adjacency)
    for t in range(T):
        vec = adjacency[t][iu].copy()
        edges = np.flatnonzero(vec)
        k = int(round(fraction * len(edges)))
        if k:
            rng = np.random.default_rng([seed, 9001, t])
            vec[rng.choice(edges, k, replace=False)] = False
        out[t] = symmetric_from_upper(vec, n)
    return out


def retrain_frozen_gcn(obs: ObservableData, splits: TemporalSplits, cfg: dict, seed: int):
    """Reconstruct the exact v2 GCN and standardized inputs for counterfactual prediction."""
    ecfg = cfg["experts"]
    train = splits.decision_times("expert_train")
    val = splits.decision_times("expert_val")
    y = label_array(obs, splits.horizon)
    z = standardise(build_inputs(obs, "base"), train)
    adjacency = mean_aggregation(obs.A_obs)
    best = None
    gcn_spec_index = 2  # mlp=0, cross_mlp=1, gcn=2 in train_all_experts
    for weight_index, weight_decay in enumerate(ecfg["weight_decay_grid"]):
        model_seed = seed * 100_000 + gcn_spec_index * 100 + weight_index
        model, val_loss, epoch = train_graphnet(
            z, adjacency, y, train, val, ecfg, weight_decay, model_seed
        )
        if best is None or val_loss < best[0]:
            best = val_loss, weight_decay, model, epoch
    return best[2], z, {"weight_decay": best[1], "val_mse": best[0], "best_epoch": best[3]}


def graph_counterfactual_predictions(obs, model, z, seed):
    """Base, edge-drop, and lag-adjacency predictions using only current/past observables."""
    all_times = np.arange(obs.T)
    z_tensor = torch.from_numpy(z)
    base_a = mean_aggregation(obs.A_obs)
    drop_a = mean_aggregation(_drop_edges(obs.A_obs, 0.20, seed))
    lag_graph = np.concatenate([obs.A_obs[:1], obs.A_obs[:-1]], axis=0)
    lag_a = mean_aggregation(lag_graph)
    return {
        "base": _predict(model, z_tensor, torch.from_numpy(base_a), all_times),
        "drop20": _predict(model, z_tensor, torch.from_numpy(drop_a), all_times),
        "lag1": _predict(model, z_tensor, torch.from_numpy(lag_a), all_times),
    }


def message_influence_stability(counterfactuals: dict[str, np.ndarray]):
    """F3: cross-sectional sensitivity of GCN predictions to observable graph perturbations."""
    base = counterfactuals["base"].astype(float)
    names, columns = [], []
    effects = []
    for perturbation in ("drop20", "lag1"):
        delta = counterfactuals[perturbation].astype(float) - base
        effects.append(delta)
        stats = np.stack([
            delta.mean(1), np.abs(delta).mean(1),
            np.sqrt((delta**2).mean(1)), delta.std(1),
            np.quantile(np.abs(delta), 0.90, axis=1),
        ], 1)
        for j, label in enumerate(("signed_mean", "mean_abs", "rms", "xs_std", "abs_q90")):
            names.append(f"F3_{perturbation}_{label}")
            columns.append(stats[:, j])
    agreement = np.sign(effects[0]) == np.sign(effects[1])
    names.extend(("F3_perturbation_sign_agreement", "F3_perturbation_effect_corr"))
    columns.append(agreement.mean(1))
    corr = np.sum(effects[0] * effects[1], 1) / np.sqrt(
        np.maximum(np.sum(effects[0] ** 2, 1) * np.sum(effects[1] ** 2, 1), 1e-12)
    )
    columns.append(corr)
    return _stack(names, columns)


def build_evidence_families(obs, preds, splits, counterfactuals=None):
    families = {
        "F1": local_residual_reliability(obs, preds, splits),
        "F2": conditional_error_covariance(obs, preds, splits),
        "F4": disagreement_distribution(obs, preds, splits),
    }
    if counterfactuals is not None:
        families["F3"] = message_influence_stability(counterfactuals)
    return families
