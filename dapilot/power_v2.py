"""Power analysis v2: inference validity for the ladder `edge` step (configs/power_ladder_v2.yaml).

Compares circular block bootstrap (several block lengths) with a placebo-rank randomisation test
that compares the retrained real-graph expert against K retrained L3 (degree + same-industry
count) null experts. Each replicate is seeded by (seed, cell, rep) so replicate ranges can run in
separate processes and still be identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .power import block_bootstrap_lower, replicate


def rank_test_p(l4_mean, l3_means):
    """One-sided p: real graph's test loss is lower than the K null-graph losses.
    l4_mean: (E,), l3_means: (K, E)."""
    K = l3_means.shape[0]
    return (1 + (l3_means <= l4_mean[None, :]).sum(0)) / (K + 1)


def run_cell(cfg, cell_name, rep_range, out: Path, log=print):
    cells = list(cfg["cells"])
    cell = cfg["cells"][cell_name]
    d_out = out / "raw" / cell_name
    d_out.mkdir(parents=True, exist_ok=True)
    for r in range(*rep_range):
        f = d_out / f"rep{r:03d}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng([cfg["experiment"]["seed"], cells.index(cell_name), r])
        d, ic, dg, raw = replicate(cell, cfg, rng, return_raw=True)
        np.savez_compressed(f, **{f"d_{k}": v for k, v in d.items()}, ic=ic,
                            L4=raw["L4_real"], L3=raw["L3_industry"], diag=json.dumps(dg))
        log(f"{cell_name} rep {r} done")


def evaluate(cfg, out: Path):
    inf = cfg["inference"]
    lvl = inf["level"]
    eff = cfg["effects_r2"]
    rows = []
    for cell_name in cfg["cells"]:
        files = sorted((out / "raw" / cell_name).glob("rep*.npz"))
        for f in files:
            z = np.load(f)
            r = int(f.stem[3:])
            rng = np.random.default_rng([cfg["experiment"]["seed"], 999, r])
            for q in ("edge", "D1"):
                dm = z[f"d_{q}"]
                for L in inf["block_lengths"]:
                    mean, lo, _ = block_bootstrap_lower(dm, L, inf["repetitions"], 1 - 2 * lvl, rng)
                    for e, r2 in enumerate(eff):
                        rows.append({"cell": cell_name, "rep": r, "quantity": q, "method": f"bootstrap_L{L}",
                                     "effect_r2": r2, "implied_ic": z["ic"][e], "mean": mean[e], "detect": lo[e] > 0})
            p = rank_test_p(z["L4"].mean(0), z["L3"].mean(1))
            for e, r2 in enumerate(eff):
                rows.append({"cell": cell_name, "rep": r, "quantity": "edge", "method": "placebo_rank",
                             "effect_r2": r2, "implied_ic": z["ic"][e], "mean": float(z["d_edge"][:, e].mean()),
                             "p": p[e], "detect": p[e] <= lvl})
    df = pd.DataFrame(rows)
    df.to_csv(out / "evaluations.csv", index=False)
    return df


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return np.nan
    ph = k / n
    return (ph + z * z / (2 * n) + z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def summarise(df, cfg, out: Path):
    eff = cfg["effects_r2"]
    pw = df.groupby(["quantity", "method", "cell", "effect_r2"]).agg(
        power=("detect", "mean"), n=("detect", "size"), implied_ic=("implied_ic", "mean")).reset_index()
    pw.to_csv(out / "power.csv", index=False)
    size = df[df.effect_r2 == 0].groupby(["quantity", "method"]).detect.agg(["sum", "size"]).reset_index()
    size["size"] = size["sum"] / size["size"].astype(float)
    size = df[df.effect_r2 == 0].groupby(["quantity", "method"]).detect.agg(k="sum", n="size").reset_index()
    size["size"] = size.k / size.n
    size["wilson_upper"] = [wilson_upper(k, n) for k, n in zip(size.k, size.n)]
    size["valid"] = (size["size"] <= 0.06) & (size.wilson_upper <= 0.08)
    rsr = ["rsr_nasdaq", "rsr_nyse", "rsr_pooled"]
    p1 = pw[(pw.effect_r2 == 1e-4) & pw.cell.isin(rsr)].groupby(["quantity", "method"]).power.mean()
    size["power_1e-4_rsr"] = [p1.get((q, m), np.nan) for q, m in zip(size.quantity, size.method)]
    size.to_csv(out / "size_and_selection.csv", index=False)

    ic = pw[pw.quantity == "edge"].groupby("effect_r2").implied_ic.mean()
    lines = ["# Power analysis v2 — inference validity for the ladder `edge` step", "",
             "Pre-registered in `configs/power_ladder_v2.yaml`. One-sided level 5% for every method.",
             "Ridge expert knows the true functional form; powers are optimistic upper bounds.", "",
             "Implied daily IC: " + ", ".join(f"R² {e:g} → {ic.loc[e]:.3f}" for e in eff), "",
             "## Size at R² = 0 (all cells pooled) and pre-registered selection", "",
             "| quantity | method | false positives | size | Wilson 95% upper | valid | mean RSR power at R² 1e-4 |",
             "|---|---|---|---|---|---|---|"]
    for r in size.rename(columns={"power_1e-4_rsr": "p_rsr"}).sort_values(["quantity", "method"]).itertuples():
        lines.append(f"| {r.quantity} | {r.method} | {int(r.k)}/{int(r.n)} | {r.size:.3f} | {r.wilson_upper:.3f} | "
                     f"{'yes' if r.valid else 'no'} | {r.p_rsr:.2f} |")
    lines.append("")
    for q in ("edge", "D1"):
        cand = size[(size.quantity == q) & size.valid]
        if len(cand):
            best = cand.sort_values("power_1e-4_rsr", ascending=False).iloc[0]
            lines.append(f"**Selected for `{q}`: {best.method}** (size {best['size']:.3f}, RSR power at 1e-4 {best['power_1e-4_rsr']:.2f}).")
        else:
            lines.append(f"**`{q}`: no method meets the validity rule.**")
    lines.append("")
    lines += ["## Size by cell at R² = 0", "", "| quantity | method | " + " | ".join(cfg["cells"]) + " |",
              "|---" * (len(cfg["cells"]) + 2) + "|"]
    s0 = pw[pw.effect_r2 == 0]
    for (q, m), g in s0.groupby(["quantity", "method"]):
        g = g.set_index("cell")
        lines.append(f"| {q} | {m} | " + " | ".join(f"{g.loc[c, 'power']:.2f} ({int(g.loc[c, 'n'])})" if c in g.index else "" for c in cfg["cells"]) + " |")
    lines.append("")
    for q in ("edge", "D1"):
        lines += [f"## Power: `{q}`", "", "| method | cell | " + " | ".join(f"{e:g}" for e in eff) + " |",
                  "|---" * (len(eff) + 2) + "|"]
        for (m, c), g in pw[pw.quantity == q].groupby(["method", "cell"], sort=False):
            g = g.set_index("effect_r2")
            lines.append(f"| {m} | {c} | " + " | ".join(f"{g.loc[e, 'power']:.2f}" for e in eff) + " |")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))
