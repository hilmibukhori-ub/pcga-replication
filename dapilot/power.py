"""Power analysis for the placebo ladder at RSR / CRSP panel sizes (see configs/power_ladder.yaml).

Large sparse panels, closed-form ridge with shared Gram matrices, common random numbers across
effect sizes (same graph, placebos and noise; only the signal coefficient changes).

L3 at this scale preserves every node's degree AND its number of same-industry neighbours:
intra-industry edges are swapped inside their industry, inter-industry edges are swapped among
inter-industry edges. (Preserving the full per-node per-industry count vector, as in
dapilot/ladder.py, leaves almost every inter-industry edge unswappable when industries have
~10 members, so that null would not randomise them.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

LEVELS = ("L0_none", "L1_er", "L2_degree", "L3_industry", "L4_real")
QUANTITIES = ("D1", "density", "degree", "industry", "edge")


# ----------------------------------------------------------------------------- graphs
def sample_graph(rng, n, block_size, intra_degree, inter_degree):
    blocks = np.arange(n) // block_size
    edges = set()
    p_intra = intra_degree / (block_size - 1)
    for b in range(blocks.max() + 1):
        idx = np.flatnonzero(blocks == b)
        iu = np.triu_indices(len(idx), 1)
        keep = rng.random(len(iu[0])) < p_intra
        edges.update(zip(idx[iu[0][keep]].tolist(), idx[iu[1][keep]].tolist()))
    m_inter = int(round(n * inter_degree / 2))
    while m_inter > 0:
        u, v = rng.integers(0, n, 2)
        if blocks[u] != blocks[v]:
            e = (min(u, v), max(u, v))
            if e not in edges:
                edges.add(e)
                m_inter -= 1
    return np.array(sorted(edges)), blocks


def er_edges(rng, n, m):
    edges = set()
    while len(edges) < m:
        u, v = rng.integers(0, n, 2)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return np.array(sorted(edges))


def _swap(E, rng, n_swaps, ok):
    """Double-edge swaps on the rows of E (in place on a copy); `ok(a, b, c, d)` vetoes swaps."""
    E = E.copy()
    S = set(map(tuple, E.tolist()))
    m = len(E)
    if m < 2:
        return E
    done = tries = 0
    while done < n_swaps and tries < 20 * n_swaps:
        tries += 1
        i, j = rng.integers(0, m, 2)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if n1 in S or n2 in S or not ok(a, b, c, d):
            continue
        S.discard((min(a, b), max(a, b)))
        S.discard((min(c, d), max(c, d)))
        S.add(n1)
        S.add(n2)
        E[i], E[j] = n1, n2
        done += 1
    return E


def degree_edges(E, rng, swaps_per_edge=10):
    return _swap(E, rng, swaps_per_edge * len(E), lambda *a: True)


def industry_edges(E, blocks, rng, swaps_per_edge=10):
    intra = blocks[E[:, 0]] == blocks[E[:, 1]]
    out = [_swap(E[~intra], rng, swaps_per_edge * int((~intra).sum()),
                 lambda a, b, c, d: blocks[a] != blocks[d] and blocks[c] != blocks[b])]
    Ei = E[intra]
    for b in np.unique(blocks[Ei[:, 0]]):
        sub = Ei[blocks[Ei[:, 0]] == b]
        out.append(_swap(sub, rng, swaps_per_edge * len(sub), lambda *a: True))
    return np.concatenate(out)


def adjacency(E, n):
    A = sp.coo_matrix((np.ones(len(E), np.float32), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    return (A + A.T).tocsr()


def same_industry_counts(E, blocks, n):
    intra = blocks[E[:, 0]] == blocks[E[:, 1]]
    return np.bincount(E[intra].ravel(), minlength=n)


# ----------------------------------------------------------------------------- features / fits
def graph_features(x, A):
    """x: (T, n, d) -> (T, n, 3d): own, neighbour mean, neighbour sum / mean degree."""
    T, n, d = x.shape
    deg = np.asarray(A.sum(1)).ravel()
    M = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    X = x.transpose(1, 0, 2).reshape(n, T * d)
    mean_agg = (M @ X).reshape(n, T, d).transpose(1, 0, 2)
    sum_agg = (A @ X).reshape(n, T, d).transpose(1, 0, 2) / max(deg.mean(), 1e-9)
    return np.concatenate([x, mean_agg, sum_agg], -1).astype(np.float32)


def fit_daily_losses(F, Y, sl, alphas):
    """Ridge per effect column of Y (T, n, E). alpha tuned on val per column.
    Returns test daily cross-sectional MSE, shape (T_test, E)."""
    k = F.shape[-1]
    Ftr = F[sl["train"]].reshape(-1, k).astype(np.float64)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Ztr = (Ftr - mu) / sd
    Ytr = Y[sl["train"]].reshape(-1, Y.shape[-1]).astype(np.float64)
    ym = Ytr.mean(0)
    G = Ztr.T @ Ztr
    ZY = Ztr.T @ (Ytr - ym)
    del Ftr, Ztr, Ytr
    Zva = ((F[sl["val"]].reshape(-1, k) - mu) / sd)
    Yva = Y[sl["val"]].reshape(-1, Y.shape[-1])
    best_v = np.full(Y.shape[-1], np.inf)
    best_b = np.zeros((k, Y.shape[-1]))
    for a in alphas:
        beta = np.linalg.solve(G + a * np.eye(k), ZY)
        v = ((Zva @ beta + ym - Yva) ** 2).mean(0)
        better = v < best_v
        best_v[better] = v[better]
        best_b[:, better] = beta[:, better]
    Fte = F[sl["test"]]
    Tt, n = Fte.shape[:2]
    Zte = (Fte.reshape(-1, k) - mu) / sd
    pred = (Zte @ best_b + ym).reshape(Tt, n, -1)
    return ((pred - Y[sl["test"]]) ** 2).mean(1)


# ----------------------------------------------------------------------------- one replicate
def replicate(cell, cfg, rng, return_raw=False):
    w = cfg["world"]
    n = cell["n"]
    T = cell["train"] + cell["val"] + cell["test"]
    # contiguous splits; one purged day at the end of train and of val
    a, b = cell["train"], cell["train"] + cell["val"]
    sl = {"train": slice(0, a - 1), "val": slice(a, b - 1), "test": slice(b, T)}

    E, blocks = sample_graph(rng, n, w["block_size"], w["intra_degree"], w["inter_degree"])
    A = adjacency(E, n)
    ex = cfg["experiment"]
    kl = ex.get("placebos_by_level") or {lev: ex["placebos_per_level"] for lev in LEVELS[1:4]}
    plac = {
        "L1_er": [adjacency(er_edges(rng, n, len(E)), n) for _ in range(kl["L1_er"])],
        "L2_degree": [adjacency(degree_edges(E, rng), n) for _ in range(kl["L2_degree"])],
        "L3_industry": [],
    }
    l3_check = []
    for _ in range(kl["L3_industry"]):
        E3 = industry_edges(E, blocks, rng)
        l3_check.append(int(np.abs(same_industry_counts(E3, blocks, n) - same_industry_counts(E, blocks, n)).sum()))
        plac["L3_industry"].append(adjacency(E3, n))

    x = rng.standard_normal((T, n, 2)).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    M = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    rel = (M @ x[:, :, 1].T).T                                  # neighbour mean of x2, (T, n)
    rel -= rel.mean(1, keepdims=True)
    own = x[:, :, 0] - x[:, :, 0].mean(1, keepdims=True)

    # noise: industry shock + fat-tailed idiosyncratic, scaled by clustered daily volatility
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
    Y = (b_own * own + noise)[:, :, None] + rel[:, :, None] * b_rel[None, None, :]
    Y = Y.astype(np.float32)

    te = sl["test"]
    ic = []
    for e, bb in enumerate(b_rel):
        sig, yy = rel[te] * bb, Y[te][:, :, e]
        if bb == 0:
            ic.append(0.0)
            continue
        sc = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
        yc = (yy - yy.mean(1, keepdims=True)) / yy.std(1, keepdims=True)
        ic.append(float((sc * yc).mean()))

    alphas = cfg["model"]["ridge_alpha_grid"]
    losses = {"L0_none": fit_daily_losses(x, Y, sl, alphas),
              "L4_real": fit_daily_losses(graph_features(x, A), Y, sl, alphas)}
    per_placebo = {}
    for lev, graphs in plac.items():
        per_placebo[lev] = np.stack([fit_daily_losses(graph_features(x, B), Y, sl, alphas) for B in graphs])
        losses[lev] = per_placebo[lev].mean(0)
    d = {
        "D1": losses["L0_none"] - losses["L4_real"],
        "density": losses["L0_none"] - losses["L1_er"],
        "degree": losses["L1_er"] - losses["L2_degree"],
        "industry": losses["L2_degree"] - losses["L3_industry"],
        "edge": losses["L3_industry"] - losses["L4_real"],
    }
    l3_overlap = float(np.mean([(B.multiply(A)).sum() / A.sum() for B in plac["L3_industry"]]))
    if return_raw:
        return d, np.array(ic), {"edges": len(E), "l3_intra_count_change": max(l3_check), "l3_overlap": l3_overlap}, \
            {"L4_real": losses["L4_real"], "L3_industry": per_placebo["L3_industry"]}
    return d, np.array(ic), {"edges": len(E), "l3_intra_count_change": max(l3_check), "l3_overlap": l3_overlap}


def block_bootstrap_lower(dmat, L, R, ci, rng):
    """dmat: (T, E). Circular moving-block bootstrap of the mean; returns (mean, lower, upper) per column."""
    T = dmat.shape[0]
    ext = np.concatenate([dmat, dmat[: L - 1]], 0)
    cs = np.concatenate([np.zeros((1, dmat.shape[1])), np.cumsum(ext, 0)], 0)
    bm = (cs[L:L + T] - cs[:T]) / L                         # mean of block starting at s
    nb = max(1, int(round(T / L)))
    starts = rng.integers(0, T, (R, nb))
    boots = bm[starts].mean(1)                              # (R, E)
    q = (1 - ci) / 2
    return dmat.mean(0), np.quantile(boots, q, 0), np.quantile(boots, 1 - q, 0)


def run(cfg, out: Path, cells=None, reps=None, log=print):
    out.mkdir(parents=True, exist_ok=True)
    inf = cfg["inference"]
    reps = reps or cfg["experiment"]["reps"]
    rows, diag = [], []
    for ci_, (name, cell) in enumerate(cfg["cells"].items()):
        if cells and name not in cells:
            continue
        rng = np.random.default_rng([cfg["experiment"]["seed"], ci_])
        t0 = time.time()
        for r in range(reps):
            d, ic, dg = replicate(cell, cfg, rng)
            diag.append({"cell": name, "rep": r, **dg})
            for q, dm in d.items():
                mean, lo, hi = block_bootstrap_lower(dm, inf["block_length_days"], inf["repetitions"], inf["ci"], rng)
                for e, r2 in enumerate(cfg["effects_r2"]):
                    rows.append({"cell": name, "rep": r, "quantity": q, "effect_r2": r2, "implied_ic": ic[e],
                                 "mean": mean[e], "ci_low": lo[e], "ci_high": hi[e]})
            log(f"{name} rep {r + 1}/{reps}  {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out / "replicates.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "placebo_diagnostics.csv", index=False)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    summarise(df, pd.DataFrame(diag), cfg, out)
    return df


def summarise(df, diag, cfg, out):
    df = df.assign(detect=df.ci_low > 0, false_neg_sign=df.ci_high < 0)
    pw = df.groupby(["cell", "quantity", "effect_r2"]).agg(
        power=("detect", "mean"), wrong_sign=("false_neg_sign", "mean"),
        implied_ic=("implied_ic", "mean"), mean_effect=("mean", "mean")).reset_index()
    pw.to_csv(out / "power.csv", index=False)
    target = cfg["inference"]["power_target"]

    def mde(sub):
        hit = sub[(sub.effect_r2 > 0) & (sub.power >= target)]
        return float(hit.effect_r2.min()) if len(hit) else np.nan

    m = pw.groupby(["cell", "quantity"]).apply(mde, include_groups=False).unstack()
    m.to_csv(out / "mde80.csv")
    eff = cfg["effects_r2"]
    lines = ["# Power analysis — placebo ladder at RSR / CRSP sizes", "",
             "Pre-registered in `configs/power_ladder.yaml`. Power = share of replicates whose 95%",
             "circular block-bootstrap CI (20-day blocks) lies above 0. Ridge expert knows the true",
             "functional form and no Holm correction is applied, so these are OPTIMISTIC upper bounds.", ""]
    ic_row = pw[(pw.quantity == "edge")].groupby("effect_r2").implied_ic.mean()
    lines += ["Implied daily cross-sectional IC of the graph signal: " +
              ", ".join(f"R² {e:g} → IC {ic_row.loc[e]:.3f}" for e in eff), ""]
    for q in ("D1", "edge", "industry", "degree", "density"):
        lines += [f"## Power: `{q}`", "", "| cell | " + " | ".join(f"{e:g}" for e in eff) + " |",
                  "|---" * (len(eff) + 1) + "|"]
        for c in cfg["cells"]:
            sub = pw[(pw.cell == c) & (pw.quantity == q)].set_index("effect_r2")
            if len(sub):
                lines.append(f"| {c} | " + " | ".join(f"{sub.loc[e, 'power']:.2f}" for e in eff) + " |")
        lines.append("")
    lines += ["## MDE80 (smallest tested R² with power ≥ 0.80; NaN = none within grid)", "",
              m.to_markdown(floatfmt="g"), ""]
    if "rsr_nasdaq" in m.index:
        rsr = m.loc[[c for c in ("rsr_nasdaq", "rsr_nyse", "rsr_pooled") if c in m.index], "edge"]
        best = rsr.min()
        ok = np.isfinite(best) and best <= 5e-4
        lines += ["## Pre-registered decision", "",
                  f"Best RSR MDE80 for `edge`: {best:g}. Rule: proceed to RSR edge-level tests only if ≤ 5e-4.",
                  f"**Decision: {'RSR can test edge-level claims' if ok else 'RSR is underpowered for edge-level claims; replication/sanity track only'}.**", ""]
    lines += ["## Placebo checks", "",
              f"- max change in same-industry neighbour counts under L3: {int(diag.l3_intra_count_change.max())} (must be 0)",
              "- L3 edge overlap with the real graph (mean by cell): " +
              ", ".join(f"{c} {v:.2f}" for c, v in diag.groupby('cell').l3_overlap.mean().items()), ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def load_config(path):
    return yaml.safe_load(Path(path).read_text())
