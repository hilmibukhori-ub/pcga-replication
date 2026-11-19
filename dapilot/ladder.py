"""Placebo ladder: nested retrained null graphs that attribute graph value to a structural level.

Light synthetic screen. Experts are closed-form ridge models on own features plus graph-aggregated
features, so a full 20-seed run takes ~6 minutes on a laptop CPU. See configs/placebo_ladder.yaml for the pre-registration.

Ladder (every placebo is fixed over time and the expert is retrained on it):
    L0 none -> L1 ER (edge count) -> L2 per-node degree -> L3 per-node degree per sector block
    -> L4 observed graph.
Steps (loss differences, positive = the added structure helps) telescope to D1 = L0 - L4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .pcga_l import decompose_loss_levels
from .stats import holm, paired_summary, verdict

LEVELS = ("L0_none", "L1_er", "L2_degree", "L3_block", "L4_real")
STEPS = ("density", "degree", "block", "edge")
SCENARIOS = ("edge", "block", "degree", "null")
PREDICTED_STEP = {"edge": "edge", "block": "block", "degree": "degree"}


# ----------------------------------------------------------------------------- graphs
def sample_graph(rng, sectors, p_intra, p_inter) -> np.ndarray:
    same = sectors[:, None] == sectors[None, :]
    P = np.where(same, p_intra, p_inter)
    U = np.triu(rng.random(P.shape) < P, 1)
    return U | U.T


def edge_list(A):
    i, j = np.nonzero(np.triu(A, 1))
    return np.stack([i, j], 1)


def from_edges(E, n):
    A = np.zeros((n, n), dtype=bool)
    A[E[:, 0], E[:, 1]] = True
    return A | A.T


def er_graph(A, rng) -> np.ndarray:
    """Erdos-Renyi graph with exactly the same number of edges."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    m = int(np.triu(A, 1).sum())
    vec = np.zeros(len(iu[0]), dtype=bool)
    vec[rng.choice(len(vec), m, replace=False)] = True
    B = np.zeros((n, n), dtype=bool)
    B[iu] = vec
    return B | B.T


def double_edge_swap(A, rng, n_swaps, blocks=None) -> np.ndarray:
    """Degree-preserving rewiring. With `blocks`, a swap (a,b),(c,d) -> (a,d),(c,b) is accepted
    only if block[a]==block[c] and block[b]==block[d], which also preserves every node's number
    of edges into every block (a degree-corrected block-model null)."""
    n = A.shape[0]
    E = edge_list(A)
    adj = A.copy()
    m = len(E)
    done = tries = 0
    while done < n_swaps and tries < 50 * n_swaps:
        tries += 1
        e1, e2 = rng.integers(0, m, 2)
        if e1 == e2:
            continue
        a, b = E[e1]
        c, d = E[e2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or adj[a, d] or adj[c, b]:
            continue
        if blocks is not None and (blocks[a] != blocks[c] or blocks[b] != blocks[d]):
            continue
        adj[a, b] = adj[b, a] = adj[c, d] = adj[d, c] = False
        adj[a, d] = adj[d, a] = adj[c, b] = adj[b, c] = True
        E[e1] = (a, d)
        E[e2] = (c, b)
        done += 1
    return adj


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def block_counts(A, blocks):
    """Per-node number of neighbours in each block, shape (n, n_blocks)."""
    k = blocks.max() + 1
    return np.stack([A[:, blocks == b].sum(1) for b in range(k)], 1)


# ----------------------------------------------------------------------------- world
def generate(cfg, seed):
    w = cfg["world"]
    rng = np.random.default_rng(seed)
    sectors = np.repeat(np.arange(w["n_sectors"]), w["per_sector"])
    n, T = len(sectors), w["T"]
    A = sample_graph(rng, sectors, w["p_intra"], w["p_inter"])
    f = rng.normal(size=(T, 1))
    x1 = rng.normal(size=(T, n))
    x2 = f + rng.normal(size=(T, n))
    common = rng.normal(size=(T, 1))
    eps = rng.normal(scale=w["noise_sd"], size=(T, n))

    deg = A.sum(1).astype(float)
    M = A / np.maximum(deg, 1)[:, None]
    same = (sectors[:, None] == sectors[None, :]) & ~np.eye(n, dtype=bool)
    S = same / same.sum(1, keepdims=True)
    rel_parts = {
        "edge": x2 @ M.T,
        "block": x2 @ S.T,
        "degree": (deg / deg.mean())[None, :] * f,
        "null": np.zeros((T, n)),
    }
    own = w["own_coef"] * x1
    worlds = {}
    for sc in SCENARIOS:
        mu = own + w["rel_coef"][sc] * rel_parts[sc]
        r_next = np.empty_like(mu)
        r_next[:-1] = mu[:-1] + common[1:] + eps[1:]   # r_{t+1} = mu_t + shocks at t+1
        r_next[-1] = 0.0                                    # no label at T-1; never used (purged)
        y = r_next - r_next.mean(1, keepdims=True)          # excess target
        mu_ex = mu - mu.mean(1, keepdims=True)
        own_ex = own - own.mean(1, keepdims=True)
        worlds[sc] = {"y": y, "mu": mu_ex, "own_mu": own_ex}
    return {"A": A, "sectors": sectors, "x": np.stack([x1, x2], -1), "worlds": worlds}


def placebo_set(A, sectors, rng, k):
    m = int(np.triu(A, 1).sum())
    out = {
        "L1_er": [er_graph(A, rng) for _ in range(k)],
        "L2_degree": [double_edge_swap(A, rng, 10 * m) for _ in range(k)],
        "L3_block": [double_edge_swap(A, rng, 10 * m, blocks=sectors) for _ in range(k)],
        "perm": [],
    }
    for _ in range(k):
        p = derangement(A.shape[0], rng)
        out["perm"].append(A[p][:, p])
    return out


# ----------------------------------------------------------------------------- experts
def graph_features(x, A):
    """[own, mean-aggregated, sum-aggregated / mean degree] for every feature; x: (T, n, d)."""
    deg = A.sum(1).astype(float)
    M = A / np.maximum(deg, 1)[:, None]
    Ssum = A / max(deg.mean(), 1e-9)
    return np.concatenate([x, np.einsum("ij,tjd->tid", M, x), np.einsum("ij,tjd->tid", Ssum, x)], -1)


def _ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ beta + ym


def fit_eval(F, y, cfg):
    """Tune alpha on val, fit on train, return test MSE (per-date mean) of the excess target."""
    sp = cfg["splits"]
    sl = {k: slice(v[0], v[1] - 1) for k, v in sp.items()}   # purge last index of each split
    flat = lambda s: (F[s].reshape(-1, F.shape[-1]), y[s].reshape(-1))
    Xtr, ytr = flat(sl["train"])
    Xva, yva = flat(sl["val"])
    best = None
    for a in cfg["model"]["ridge_alpha_grid"]:
        pred = _ridge_fit(Xtr, ytr, a)
        v = np.mean((pred(Xva) - yva) ** 2)
        if best is None or v < best[0]:
            best = (v, pred)
    Xte, yte = flat(sl["test"])
    return float(np.mean((best[1](Xte) - yte) ** 2))


# ----------------------------------------------------------------------------- run
def run_seed(cfg, seed):
    g = generate(cfg, seed)
    rng = np.random.default_rng(10_000 + seed)
    plac = placebo_set(g["A"], g["sectors"], rng, cfg["experiment"]["placebos_per_level"])
    x = g["x"]
    feats = {"L0_none": [x], "L4_real": [graph_features(x, g["A"])]}
    for lev in ("L1_er", "L2_degree", "L3_block", "perm"):
        feats[lev] = [graph_features(x, B) for B in plac[lev]]
    te = slice(*cfg["splits"]["test"])
    rows = []
    for sc, wd in g["worlds"].items():
        row = {"seed": seed, "scenario": sc}
        for lev, Fs in feats.items():
            row[lev] = float(np.mean([fit_eval(F, wd["y"], cfg) for F in Fs]))
        y_te = wd["y"][te][:-1]
        row["oracle_full"] = float(np.mean((wd["mu"][te][:-1] - y_te) ** 2))
        row["oracle_own"] = float(np.mean((wd["own_mu"][te][:-1] - y_te) ** 2))
        rows.append(row)
    diag = {
        "seed": seed,
        "edges": int(np.triu(g["A"], 1).sum()),
        "L2_block_count_change": float(np.mean([np.abs(block_counts(B, g["sectors"]) - block_counts(g["A"], g["sectors"])).sum() for B in plac["L2_degree"]])),
        "L3_block_count_change": float(np.mean([np.abs(block_counts(B, g["sectors"]) - block_counts(g["A"], g["sectors"])).sum() for B in plac["L3_block"]])),
        "L3_edge_overlap": float(np.mean([(B & g["A"]).sum() / g["A"].sum() for B in plac["L3_block"]])),
    }
    return rows, diag


def add_steps(df):
    df = decompose_loss_levels(df)
    df["D2_er"] = df["L1_er"] - df["L4_real"]
    df["D2_perm"] = df["perm"] - df["L4_real"]
    df["oracle_rel_gain"] = df["oracle_own"] - df["oracle_full"]
    return df


def test_hypotheses(df, cfg):
    h = cfg["hypotheses"]
    by = {sc: d.sort_values("seed") for sc, d in df.groupby("scenario")}
    tests = [
        ("H1", "edge", "edge", "positive"),
        ("H2", "block", "block", "positive"),
        ("H3", "degree", "degree", "positive"),
        ("H5a", "block", "D2_er", "positive"),
        ("H5b", "degree", "D2_perm", "positive"),
    ] + [(f"H6_{s}", "null", s, "two-sided") for s in STEPS]
    sums = [paired_summary(by[sc][col].to_numpy()) for _, sc, col, _ in tests]
    padj = holm([s["wilcoxon_p"] for s in sums])
    out = []
    for (hid, sc, col, pred), s, pa in zip(tests, sums, padj):
        out.append({"id": hid, "scenario": sc, "quantity": col, "predicted": pred, **s,
                    "p_holm": pa, "verdict": verdict(s, pa, pred, h["alpha"])})
    for sc, step in PREDICTED_STEP.items():
        d = by[sc]
        pos = d[list(STEPS)].clip(lower=0)
        share = d[step].clip(lower=0) / pos.sum(1).replace(0, np.nan)
        frac = float((share >= h["attribution_share"]).mean())
        out.append({"id": f"H4_{sc}", "scenario": sc, "quantity": f"share({step})",
                    "predicted": f">= {h['attribution_share']} in >= {h['attribution_seed_frac']} of seeds",
                    "mean": float(share.mean()), "frac_seeds": frac,
                    "verdict": "supported" if frac >= h["attribution_seed_frac"] else "not supported"})
    return pd.DataFrame(out)


def write_report(df, hyp, diag, out):
    lines = ["# Placebo ladder v1 — synthetic screen", "",
             "Pre-registered in `configs/placebo_ladder.yaml`. Ridge experts, 20 seeds, 10 retrained",
             "placebos per level. Values are test-MSE differences x 1e3 (positive = structure helps).", ""]
    cols = ["oracle_rel_gain", "D1", *STEPS, "D2_er", "D2_perm"]
    g = df.groupby("scenario")[cols]
    m, s = g.mean() * 1e3, g.std(ddof=1) * 1e3 / np.sqrt(df.seed.nunique())
    lines += ["## Ladder decomposition (mean ± s.e. over seeds)", "",
              "| scenario | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for sc in SCENARIOS:
        lines.append(f"| {sc} | " + " | ".join(f"{m.loc[sc, c]:+.3f} ± {s.loc[sc, c]:.3f}" for c in cols) + " |")
    lines += ["", "## Pre-registered hypotheses", "",
              "| id | scenario | quantity | mean x1e3 / share | CI x1e3 | p_holm | verdict |",
              "|---|---|---|---|---|---|---|"]
    for r in hyp.itertuples():
        if r.id.startswith("H4"):
            lines.append(f"| {r.id} | {r.scenario} | {r.quantity} | {r.mean:.2f} (seeds ≥ thr: {r.frac_seeds:.0%}) | | | {r.verdict} |")
        else:
            lines.append(f"| {r.id} | {r.scenario} | {r.quantity} | {r.mean*1e3:+.3f} | [{r.t_ci_low*1e3:+.3f}, {r.t_ci_high*1e3:+.3f}] | {r.p_holm:.2g} | {r.verdict} |")
    lines += ["", "## Placebo sanity checks (mean over seeds)", "",
              f"- edges per graph: {diag.edges.mean():.1f}",
              f"- L2 (degree) total |change| in per-node block counts: {diag.L2_block_count_change.mean():.1f}",
              f"- L3 (degree+block) total |change| in per-node block counts: {diag.L3_block_count_change.mean():.1f} (must be 0)",
              f"- L3 edge overlap with the real graph: {diag.L3_edge_overlap.mean():.2f}", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def run(cfg, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows, diags = [], []
    for seed in cfg["experiment"]["seeds"]:
        r, d = run_seed(cfg, seed)
        rows += r
        diags.append(d)
    df = add_steps(pd.DataFrame(rows))
    diag = pd.DataFrame(diags)
    hyp = test_hypotheses(df, cfg)
    df.to_csv(out / "per_seed.csv", index=False)
    diag.to_csv(out / "placebo_diagnostics.csv", index=False)
    hyp.to_csv(out / "hypotheses.csv", index=False)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    write_report(df, hyp, diag, out)
    return df, hyp


def load_config(path):
    return yaml.safe_load(Path(path).read_text())
