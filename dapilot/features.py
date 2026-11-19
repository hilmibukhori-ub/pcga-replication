"""Selector information set and meta-labels.

Selector features at decision time t may use:
  * structural diagnostics of the observed graph at t (and earlier),
  * expert predictions made at t (disagreement),
  * observable market state at t,
  * realised losses of predictions whose labels have matured by t, i.e. decision times
    s with s + h <= t, and only out-of-sample predictions (s >= expert-val start).
Meta-labels (realised benefit at t) are labels that mature at t + h and are masked beyond a
cutoff so that locked-test labels are physically absent from anything selectors fit on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dgp import ObservableData
from .experts import label_array
from .splits import LeakageError, TemporalSplits

GROUPS = ("structural", "evidence", "disagreement", "market")


def realised_loss(pred: np.ndarray, y: np.ndarray, valid_from: int) -> np.ndarray:
    """L[s] = mean_i (y[s, i] - pred[s, i])^2 for out-of-sample s >= valid_from, else NaN."""
    L = ((y - pred) ** 2).mean(1)
    L[:valid_from] = np.nan
    return L


def matured_window_mean(series: np.ndarray, t: int, w: int, h: int) -> float:
    """Mean of series[s] over the w most recent decision times s whose labels matured by t."""
    hi = t - h          # last decision time whose label (at hi + h) is available at t
    lo = hi - w + 1
    if hi + h > t:
        raise LeakageError("label not yet matured")
    if lo < 0:
        return np.nan
    win = series[lo:hi + 1]
    return float(win.mean()) if np.isfinite(win).all() else np.nan


@dataclass
class SelectorFeatures:
    Z: np.ndarray                      # (T, d), NaN where unavailable
    names: list[str]
    groups: dict[str, list[int]] = field(default_factory=dict)

    def columns(self, groups) -> list[int]:
        return sorted(c for g in groups for c in self.groups[g])


def build_selector_features(
    obs: ObservableData,
    preds: dict,
    g_key: str,
    n_key: str,
    placebo_keys: list[str],
    struct: np.ndarray,
    struct_names,
    splits: TemporalSplits,
    scfg: dict,
) -> SelectorFeatures:
    h = splits.horizon
    T = obs.T
    y = label_array(obs, h)
    v0 = splits.start("expert_val")   # first out-of-sample expert prediction
    LG = realised_loss(preds[g_key], y, v0)
    LN = realised_loss(preds[n_key], y, v0)
    B = LN - LG
    if placebo_keys:
        LP = np.mean([realised_loss(preds[k], y, v0) for k in placebo_keys], 0)
        D2 = LP - LG
    else:
        D2 = np.full(T, np.nan)
    # realised cross-sectional return variance, decision-time indexed: y[s] matures at s + h
    RV = np.full(T, np.nan)
    has = np.isfinite(y).all(1)
    RV[has] = y[has].var(1)
    RV[:v0] = np.nan

    cols, names, groups = [], [], {g: [] for g in GROUPS}

    def add(group, name, values):
        groups[group].append(len(names))
        names.append(name)
        cols.append(np.asarray(values, dtype=np.float64))

    for j, n in enumerate(struct_names):
        add("structural", f"S_{n}", struct[:, j])

    aw = scfg["aux_window"]
    ev = {f"E_benefit_w{w}": (B, w) for w in scfg["evidence_windows"]}
    ev[f"E_d2_w{aw}"] = (D2, aw)
    ev[f"E_logloss_G_w{aw}"] = (np.log(LG), aw)
    ev[f"E_logloss_N_w{aw}"] = (np.log(LN), aw)
    ev[f"E_retvar_w{aw}"] = (RV, aw)
    for name, (series, w) in ev.items():
        add("evidence", name, [matured_window_mean(series, t, w, h) for t in range(T)])

    pG, pN = preds[g_key], preds[n_key]
    add("disagreement", "Q_mad_GN", np.abs(pG - pN).mean(1))
    add("disagreement", "Q_rms_GN", np.sqrt(((pG - pN) ** 2).mean(1)))
    if placebo_keys:
        pP = np.mean([preds[k] for k in placebo_keys], 0)
        add("disagreement", "Q_rms_GP", np.sqrt(((pG - pP) ** 2).mean(1)))

    for k in range(obs.factors.shape[1]):
        add("market", f"M_factor{k}", obs.factors[:, k])
    add("market", "M_feature_dispersion", obs.X.std(1).mean(-1))

    return SelectorFeatures(Z=np.stack(cols, 1), names=names, groups=groups)


@dataclass
class MetaData:
    """Everything a selector may fit on. Labels maturing at or after `cutoff` are NaN."""

    Z: np.ndarray            # standardised features (T, d)
    names: list[str]
    groups: dict
    B: np.ndarray            # realised benefit L_N - L_G        (T,)
    D2: np.ndarray           # realised placebo gap L_P - L_G    (T,)
    LG: np.ndarray
    LN: np.ndarray
    pG: np.ndarray           # (T, N)
    pN: np.ndarray
    y: np.ndarray            # (T, N) masked labels
    fit_idx: np.ndarray
    val_idx: np.ndarray
    cal_idx: np.ndarray
    cutoff: int

    def cols(self, groups) -> list[int]:
        return sorted(c for g in groups for c in self.groups[g])


def make_meta_data(
    obs: ObservableData, preds: dict, g_key, n_key, placebo_keys, feats: SelectorFeatures,
    splits: TemporalSplits,
) -> MetaData:
    h = splits.horizon
    cutoff = splits.start("test")
    y = label_array(obs, h).astype(np.float64)
    y[max(cutoff - h, 0):] = np.nan     # labels maturing in the locked test are removed
    v0 = splits.start("expert_val")
    LG, LN = realised_loss(preds[g_key], y, v0), realised_loss(preds[n_key], y, v0)
    LP = np.mean([realised_loss(preds[k], y, v0) for k in placebo_keys], 0) if placebo_keys \
        else np.full(obs.T, np.nan)

    fit_idx = splits.decision_times("meta_train")
    val_idx = splits.decision_times("selector_val")
    cal_idx = splits.decision_times("calibration")
    for idx in (fit_idx, val_idx, cal_idx):
        splits.assert_labels_before(idx, "test")
        if not np.isfinite(feats.Z[idx]).all():
            raise LeakageError("selector features unavailable (window reaches in-sample period)")
        if not np.isfinite(y[idx]).all():
            raise LeakageError("missing labels in a fitting split")

    mu = feats.Z[fit_idx].mean(0)
    sd = feats.Z[fit_idx].std(0)
    sd[sd < 1e-10] = 1.0
    Z = (feats.Z - mu) / sd
    return MetaData(
        Z=Z, names=feats.names, groups=feats.groups, B=LN - LG, D2=LP - LG, LG=LG, LN=LN,
        pG=preds[g_key].astype(np.float64), pN=preds[n_key].astype(np.float64), y=y,
        fit_idx=fit_idx, val_idx=val_idx, cal_idx=cal_idx, cutoff=cutoff,
    )
