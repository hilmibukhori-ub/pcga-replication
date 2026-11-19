"""RSR confirmatory pipeline (PREREG_RSR.md, configs/rsr_prereg.yaml; clarifications in DEVIATIONS.md).

Two modes:
  dev      - the arrays are truncated at the end of the validation year BEFORE anything is computed.
             Splits are shifted by one year: train 2013-14, val 2015, pseudo-test 2016. Used for the
             smoke test (PREREG §10 step 3). The 2017 test rows are never loaded into memory.
  confirm  - the prereg splits. Refuses to run unless the frozen files match PREREG_FREEZE.md.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .power import adjacency, degree_edges, er_edges, graph_features, industry_edges
from .power_v2 import rank_test_p
from .rsr_data import REPO, industry_blocks, tickers, wiki_edges
from .stats import holm

ROOT = Path(__file__).resolve().parents[1]
MISSING = -1234.0
PAD = 29
TEST_START = "2017-01-03"


class FreezeError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- freeze guard
def verify_freeze():
    """Every sha256 in the 'Frozen files' table of PREREG_FREEZE.md must match the file on disk."""
    txt = (ROOT / "PREREG_FREEZE.md").read_text()
    block = txt.split("## Frozen files")[1].split("##")[0]
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", block)
    if not rows:
        raise FreezeError("no frozen hashes found")
    for path, h in rows:
        got = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if got != h:
            raise FreezeError(f"{path}: sha256 {got} != frozen {h}")
    return rows


# ----------------------------------------------------------------------------- data
def trading_dates(market):
    d = [l.strip()[:10] for l in open(REPO / f"data/{market}_aver_line_dates.csv") if l.strip()]
    return d[PAD:]


def load_close(market, max_date: str):
    """(T, n) close (scaled by a per-stock constant; NaN = missing), rows with date <= max_date only."""
    d = trading_dates(market)
    tick = tickers(market)
    cols = []
    for t in tick:
        a = np.loadtxt(REPO / f"data/2013-01-01/{market}_{t}_1.csv", delimiter=",", dtype=np.float64)
        if market == "NASDAQ":
            a = a[:-1]
        cols.append(a[:, -1])
    C = np.stack(cols, 1)
    d = d[: len(C)]
    keep = int(sum(x <= max_date for x in d))
    C = C[:keep].copy()
    C[np.isclose(C, MISSING)] = np.nan
    return C, d[:keep]


def split_rows(dates, split):
    """Row index ranges per split. The last decision date of EVERY split is purged: its label matures
    on the first day of the next split (or after the sample ends, for the final split)."""
    out = {}
    for s, (a, b) in split.items():
        ii = [k for k, x in enumerate(dates) if a <= x <= b]
        if not ii:
            raise ValueError(f"split {s} empty")
        out[s] = (ii[0], ii[-1] - 1)
    return out


def dev_split(split):
    """Shift one year back: train 2013-14, val 2015, pseudo-test 2016 (= the prereg validation year)."""
    return {"train": [split["train"][0], "2014-12-31"], "val": ["2015-01-02", split["train"][1]],
            "test": list(split["val"])}


# ----------------------------------------------------------------------------- features / target
def returns(C):
    r = np.full_like(C, np.nan)
    r[1:] = C[1:] / C[:-1] - 1
    return r


def features(C, lookback=20):
    """(T, n, 4): ret_1d, ret_5d, ret_20d, realised vol over the last 20 daily returns. Uses rows <= t only.
    Missing -> daily cross-sectional median; then daily z-score. Rows < lookback are NaN."""
    T, n = C.shape
    r = returns(C)
    X = np.full((T, n, 4), np.nan)
    for t in range(lookback, T):
        X[t, :, 0] = r[t]
        X[t, :, 1] = C[t] / C[t - 5] - 1
        X[t, :, 2] = C[t] / C[t - 20] - 1
        w = r[t - 19:t + 1]
        ok = np.isfinite(w).sum(0) >= 10
        X[t, :, 3] = np.where(ok, np.nanstd(np.where(np.isfinite(w), w, np.nan), 0), np.nan)
    for t in range(lookback, T):
        for j in range(4):
            v = X[t, :, j]
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            v = np.where(np.isfinite(v), v, med)
            sd = v.std()
            X[t, :, j] = (v - v.mean()) / sd if sd > 0 else 0.0
    return X


def target(C):
    """y[t] = r[t+1] - equal-weight cross-sectional mean of r[t+1]; mask[t] = close t and t+1 observed."""
    r = returns(C)
    y = np.full_like(C, np.nan)
    y[:-1] = r[1:]
    mask = np.isfinite(y)
    m = np.nanmean(np.where(mask, y, np.nan), 1, keepdims=True)
    y = np.where(mask, y - m, 0.0)
    return y, mask


# ----------------------------------------------------------------------------- ridge
def fit_ridge_losses(F, Y, Yraw, M, rows, alphas):
    """Masked pooled ridge. rows: {'train': (lo, hi), 'val': ..., 'test': ...} inclusive.
    Returns dict of test daily losses: mse (winsorised target), mse_raw, mae; plus the chosen alpha."""
    k = F.shape[-1]

    def take(s):
        lo, hi = rows[s]
        f, y, m = F[lo:hi + 1], Y[lo:hi + 1], M[lo:hi + 1]
        return f[m], y[m]

    Ftr, Ytr = take("train")
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Z = (Ftr - mu) / sd
    ym = Ytr.mean()
    G, ZY = Z.T @ Z, Z.T @ (Ytr - ym)
    Fva, Yva = take("val")
    Zva = (Fva - mu) / sd
    best = (np.inf, None, None)
    for a in alphas:
        beta = np.linalg.solve(G + a * np.eye(k), ZY)
        v = float(((Zva @ beta + ym - Yva) ** 2).mean())
        if v < best[0]:
            best = (v, beta, a)
    _, beta, a = best
    lo, hi = rows["test"]
    Ft, Mt = F[lo:hi + 1], M[lo:hi + 1]
    pred = ((Ft - mu) / sd) @ beta + ym
    cnt = Mt.sum(1)

    def daily(e):
        return np.where(Mt, e, 0.0).sum(1) / cnt

    return {"mse": daily((pred - Y[lo:hi + 1]) ** 2), "mse_raw": daily((pred - Yraw[lo:hi + 1]) ** 2),
            "mae": daily(np.abs(pred - Yraw[lo:hi + 1])), "alpha": a, "pred": pred}


def daily_ic(pred, y, M):
    out = []
    for t in range(len(pred)):
        m = M[t]
        p, q = pred[t][m], y[t][m]
        out.append(np.corrcoef(p, q)[0, 1] if m.sum() > 2 and p.std() > 0 else np.nan)
    return np.array(out)


# ----------------------------------------------------------------------------- inference
def bootstrap_p_one_sided(d, L, R, rng):
    """Circular block bootstrap of the mean of daily series d; p = share of bootstrap means <= 0;
    also the 90% two-sided interval (PREREG §6.1)."""
    T = len(d)
    ext = np.concatenate([d, d[: L - 1]])
    cs = np.concatenate([[0.0], np.cumsum(ext)])
    bm = (cs[L:L + T] - cs[:T]) / L
    boots = bm[rng.integers(0, T, (R, max(1, round(T / L))))].mean(1)
    return float((boots <= 0).mean()), (float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95)))


# ----------------------------------------------------------------------------- one market
def run_market(cfg, market, mode, log=print, k_l3=None):
    split = cfg["data"]["split"]
    if mode == "dev":
        sp_ = dev_split(split)
        max_date = split["val"][1]
    elif mode == "confirm":
        verify_freeze()
        sp_ = split
        max_date = split["test"][1]
    else:
        raise ValueError(mode)
    C, dates = load_close(market, max_date)
    if mode == "dev":
        assert dates[-1] < TEST_START, "dev mode must not load test rows"
    rows = split_rows(dates, sp_)
    lb = cfg["task"]["min_lookback_days"]
    X = features(C, lb)
    Yraw, M = target(C)
    M[:lb] = False
    lo, hi = rows["train"]
    q = cfg["task"]["winsorise_quantiles_from_train"]
    ql, qh = np.quantile(Yraw[lo:hi + 1][M[lo:hi + 1]], q)
    Y = np.clip(Yraw, ql, qh)
    X = np.nan_to_num(X)

    E, blocks = wiki_edges(market), industry_blocks(market)
    n = C.shape[1]
    A = adjacency(E, n)
    rng = np.random.default_rng([cfg["experiment"]["seed"], ["NASDAQ", "NYSE"].index(market)])
    kp = cfg["graphs"]["placebos"]
    K3 = k_l3 or kp["L3_industry"]
    plac = {"L1_er": [adjacency(er_edges(rng, n, len(E)), n) for _ in range(kp["L1_er"])],
            "L2_degree": [adjacency(degree_edges(E, rng), n) for _ in range(kp["L2_degree"])],
            "L3_industry": [adjacency(industry_edges(E, blocks, rng), n) for _ in range(K3)]}
    alphas = cfg["experts"]["primary"]["alpha_grid"]
    fit = lambda F: fit_ridge_losses(F, Y, Yraw, M, rows, alphas)  # noqa: E731

    res = {"L0_none": fit(X), "L4_real": fit(graph_features(X, A))}
    per = {}
    for lev, gs in plac.items():
        per[lev] = [fit(graph_features(X, B)) for B in gs]
        log(f"{market} {lev}: {len(gs)} placebos fitted")
    lo, hi = rows["test"]
    out = {"market": market, "mode": mode, "test_dates": [dates[lo], dates[hi]], "test_days": hi - lo + 1,
           "winsor": [float(ql), float(qh)], "alpha": {"L0": res["L0_none"]["alpha"], "L4": res["L4_real"]["alpha"]},
           "l3_overlap": float(np.mean([(B.multiply(A)).sum() / A.sum() for B in plac["L3_industry"]]))}
    for loss in ("mse", "mse_raw", "mae"):
        Lm = {"L0_none": res["L0_none"][loss], "L4_real": res["L4_real"][loss]}
        for lev in per:
            Lm[lev] = np.mean([p[loss] for p in per[lev]], 0)
        out[loss] = {
            "D1": Lm["L0_none"] - Lm["L4_real"],
            "density": Lm["L0_none"] - Lm["L1_er"],
            "degree": Lm["L1_er"] - Lm["L2_degree"],
            "industry": Lm["L2_degree"] - Lm["L3_industry"],
            "edge": Lm["L3_industry"] - Lm["L4_real"],
            "L3_each_mean": np.array([p[loss].mean() for p in per["L3_industry"]]),
            "L4_mean": float(Lm["L4_real"].mean()),
        }
    out["ic"] = {k: float(np.nanmean(daily_ic(res[k]["pred"], Yraw[lo:hi + 1], M[lo:hi + 1]))) for k in res}
    return out


def confirmatory(cfg, results, rng):
    """R1-R4 (PREREG §8) from per-market results. Returns rows with raw and Holm p."""
    inf = cfg["inference"]
    rows = []
    for spec in cfg["confirmatory"]:
        r = results[spec["market"]]["mse"]
        if spec["quantity"] == "D1":
            p, ci = bootstrap_p_one_sided(r["D1"], inf["D1"]["block_length"], inf["D1"]["repetitions"], rng)
            est = float(r["D1"].mean())
        else:
            p = float(rank_test_p(np.array([r["L4_mean"]]), r["L3_each_mean"][:, None])[0])
            _, ci = bootstrap_p_one_sided(r["edge"], 20, inf["D1"]["repetitions"], rng)
            est = float(r["edge"].mean())
        rows.append({**spec, "estimate": est, "ci90": ci, "p": p})
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["verdict"] = "supported" if a <= inf["multiplicity"]["alpha"] else (
            "opposite direction" if r["ci90"][1] < 0 else "not detected")
    return rows
