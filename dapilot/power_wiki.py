"""P1: power for R1-R4 on the real RSR Wiki topology (configs/power_wiki_p1.yaml).

The DGP is the one in dapilot/power.py::replicate, with the sampled block graph replaced by the
fixed Wiki graph and the RSR industry labels. No RSR return data are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .power import adjacency, block_bootstrap_lower, fit_daily_losses, graph_features, industry_edges
from .power_v2 import rank_test_p, wilson_upper
from .rsr_data import industry_blocks, wiki_edges


def replicate_fixed(E, blocks, cell, cfg, rng):
    w = cfg["world"]
    n = len(blocks)
    T = cell["train"] + cell["val"] + cell["test"]
    a, b = cell["train"], cell["train"] + cell["val"]
    sl = {"train": slice(0, a - 1), "val": slice(a, b - 1), "test": slice(b, T)}
    A = adjacency(E, n)
    K = cfg["experiment"]["K_L3"]
    l3 = [industry_edges(E, blocks, rng) for _ in range(K)]
    l3_A = [adjacency(e, n) for e in l3]

    x = rng.standard_normal((T, n, 2)).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    M = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    rel = (M @ x[:, :, 1].T).T
    rel -= rel.mean(1, keepdims=True)
    own = x[:, :, 0] - x[:, :, 0].mean(1, keepdims=True)

    s = w["sector_shock_share"]
    logv = np.empty(T)
    logv[0] = 0.0
    innov = w["vol_sd"] * np.sqrt(1 - w["vol_ar"] ** 2)
    for t in range(1, T):
        logv[t] = w["vol_ar"] * logv[t - 1] + innov * rng.standard_normal()
    v = np.exp(logv - logv.mean())
    shock = rng.standard_normal((T, blocks.max() + 1))[:, blocks]
    df = w["t_df"]
    idio = rng.standard_t(df, (T, n)) / np.sqrt(df / (df - 2))
    noise = v[:, None] * (np.sqrt(s) * shock + np.sqrt(1 - s) * idio)
    noise -= noise.mean(1, keepdims=True)
    noise /= noise.std()

    effects = np.asarray(cfg["effects_r2"])
    b_rel = np.sqrt(effects / rel.var())
    b_own = np.sqrt(w["own_r2"] / own.var())
    Y = ((b_own * own + noise)[:, :, None] + rel[:, :, None] * b_rel[None, None, :]).astype(np.float32)

    te = sl["test"]
    ic = []
    for e, bb in enumerate(b_rel):
        if bb == 0:
            ic.append(0.0)
            continue
        sig, yy = rel[te] * bb, Y[te][:, :, e]
        sc = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
        yc = (yy - yy.mean(1, keepdims=True)) / yy.std(1, keepdims=True)
        ic.append(float((sc * yc).mean()))

    alphas = cfg["model"]["ridge_alpha_grid"]
    L0 = fit_daily_losses(x, Y, sl, alphas)
    L4 = fit_daily_losses(graph_features(x, A), Y, sl, alphas)
    L3 = np.stack([fit_daily_losses(graph_features(x, B), Y, sl, alphas) for B in l3_A])
    overlap = float(np.mean([(B.multiply(A)).sum() / A.sum() for B in l3_A]))
    return L0, L4, L3, np.array(ic), overlap


def run_cell(cfg, market, rep_range, out: Path, log=print):
    E, blocks = wiki_edges(market), industry_blocks(market)
    cell = cfg["cells"][market]
    d_out = out / "raw" / market
    d_out.mkdir(parents=True, exist_ok=True)
    mk = list(cfg["cells"]).index(market)
    for r in range(*rep_range):
        f = d_out / f"rep{r:03d}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng([cfg["experiment"]["seed"], mk, r])
        L0, L4, L3, ic, ov = replicate_fixed(E, blocks, cell, cfg, rng)
        np.savez_compressed(f, L0=L0, L4=L4, L3=L3, ic=ic, overlap=ov)
        log(f"{market} rep {r} done (L3 overlap {ov:.2f})")


def evaluate(cfg, out: Path):
    inf = cfg["inference"]
    eff = cfg["effects_r2"]
    rows = []
    for market in cfg["cells"]:
        for f in sorted((out / "raw" / market).glob("rep*.npz")):
            z = np.load(f)
            r = int(f.stem[3:])
            rng = np.random.default_rng([cfg["experiment"]["seed"], 999, r])
            d1 = z["L0"] - z["L4"]
            # one-sided p for D1: share of bootstrap means <= 0 (PREREG §6.1)
            T = d1.shape[0]
            L = inf["block_length"]
            ext = np.concatenate([d1, d1[: L - 1]], 0)
            cs = np.concatenate([np.zeros((1, d1.shape[1])), np.cumsum(ext, 0)], 0)
            bm = (cs[L:L + T] - cs[:T]) / L
            starts = rng.integers(0, T, (inf["repetitions"], max(1, round(T / L))))
            boots = bm[starts].mean(1)
            p_d1 = (boots <= 0).mean(0)
            p_edge = rank_test_p(z["L4"].mean(0), z["L3"].mean(1))
            for e, r2 in enumerate(eff):
                rows.append({"market": market, "rep": r, "effect_r2": r2, "implied_ic": z["ic"][e],
                             "overlap": float(z["overlap"]), "p_D1": p_d1[e], "p_edge": p_edge[e]})
    return pd.DataFrame(rows)


def summarise(df, cfg, out: Path):
    inf = cfg["inference"]
    lv, hw = inf["level"], inf["holm_worst_level"]
    g = df.groupby(["market", "effect_r2"])
    s = pd.DataFrame({
        "n": g.size(),
        "implied_ic": g.implied_ic.mean(),
        "D1_power_05": g.p_D1.apply(lambda p: (p <= lv).mean()),
        "D1_power_holm": g.p_D1.apply(lambda p: (p <= hw).mean()),
        "edge_power_05": g.p_edge.apply(lambda p: (p <= lv).mean()),
        "edge_power_holm": g.p_edge.apply(lambda p: (p <= hw).mean()),
    }).reset_index()
    s.to_csv(out / "power.csv", index=False)
    lines = ["# P1 — power on the real RSR Wiki topology", "",
             "See `configs/power_wiki_p1.yaml` and DEVIATIONS.md P1. No RSR returns used. "
             "Ridge knows the true functional form, so these are optimistic upper bounds.", "",
             f"`*_05` = rejection at 0.05; `*_holm` = rejection at {hw} (smallest Holm threshold, conservative).", "",
             "| market | R² | IC | n | D1 @.05 | D1 @Holm | edge @.05 | edge @Holm |", "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.market} | {r.effect_r2:g} | {r.implied_ic:.3f} | {r.n} | {r.D1_power_05:.2f} | "
                     f"{r.D1_power_holm:.2f} | {r.edge_power_05:.2f} | {r.edge_power_holm:.2f} |")
    lines += ["", "## Size at R² = 0 (Wilson 95% upper)", ""]
    for m, sub in df[df.effect_r2 == 0].groupby("market"):
        for q in ("p_D1", "p_edge"):
            k = int((sub[q] <= lv).sum())
            lines.append(f"- {m} {q[2:]}: {k}/{len(sub)} = {k / len(sub):.3f} (upper {wilson_upper(k, len(sub)):.3f})")
    lines += ["", "## MDE80 (smallest R² with power ≥ 0.80 at the Holm-worst level)", ""]
    for m, sub in s.groupby("market"):
        for q in ("D1_power_holm", "edge_power_holm"):
            ok = sub[(sub[q] >= 0.8) & (sub.effect_r2 > 0)]
            lines.append(f"- {m} {q.split('_')[0]}: " + (f"R² {ok.effect_r2.min():g} (IC ≈ {ok.implied_ic.iloc[0]:.3f})" if len(ok) else "none in grid"))
    ov = df.groupby("market").overlap.mean()
    lines += ["", "## L3 placebo edge overlap with Wiki (mean)", ""] + [f"- {m}: {v:.2f}" for m, v in ov.items()]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return s
