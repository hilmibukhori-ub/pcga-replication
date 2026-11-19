"""Aggregate per-seed results: tidy tables, paired hypothesis tests, plots, report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .dgp import REGIME_NAMES  # noqa: E402
from .config import config_hash  # noqa: E402
from .stats import holm, paired_summary, verdict  # noqa: E402

SUBSETS = ("all",) + REGIME_NAMES
METHODS = ("AlwaysN", "AlwaysG", "FixedEnsemble", "Gate", "CondRisk", "A1", "A2a",
           "A2a_noStruct", "A2a_noAux", "A2b", "Oracle")
REGIME_COLORS = {"informative": "#2a78d6", "no_signal": "#eb6834", "corrupted": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# (id, family, description, kind, spec, predicted sign of diff)
#   kind 'sel'  : diff = metric(slot, A, subset) - metric(slot, B, subset)
#   kind 'gv'   : value = graph_value[subset][key]            (one-sample vs 0)
#   kind 'gvd'  : diff = graph_value[s1][key] - graph_value[s2][key]
#   kind 'expert_slot': raw expert-quality gap, placebo AlwaysG risk - real AlwaysG risk
#   kind 'selector_gain': (AlwaysG risk - selector risk) in real minus the same gain in placebo
HYPOTHESES = [
    ("HD1", "D", "Graph value D1 (GCN vs matched MLP) > 0 in informative regime",
     "gv", ("informative", "D1_realised_vs_mlp"), "positive"),
    ("HD2", "D", "D1 < 0 in no-signal regime (graph hurts without relational signal)",
     "gv", ("no_signal", "D1_realised_vs_mlp"), "negative"),
    ("HD3", "D", "D1 lower under corrupted adjacency than informative",
     "gvd", ("corrupted", "informative", "D1_realised_vs_mlp"), "negative"),
    ("HD4", "D", "D2 (placebo minus real graph) > 0 in informative regime",
     "gv", ("informative", "D2_realised"), "positive"),
    ("HD5", "D", "D2 lower under corrupted adjacency than informative",
     "gvd", ("corrupted", "informative", "D2_realised"), "negative"),
    ("HD6", "D", "GCN beats cross-asset non-graph control in informative regime",
     "gv", ("informative", "D1_realised_vs_cross_mlp"), "positive"),
    ("HD7", "D", "Real GCN has lower conditional risk than a retrained placebo GCN",
     "expert_slot", ("all",), "positive"),
    ("HA1", "A", "A2a regret < A1 regret", "sel", ("real", "A2a", "A1", "all"), "negative"),
    ("HA2", "A", "A2a regret < A2a without structural diagnostics",
     "sel", ("real", "A2a", "A2a_noStruct", "all"), "negative"),
    ("HA3", "A", "A2a regret < A2a without D2 auxiliary task",
     "sel", ("real", "A2a", "A2a_noAux", "all"), "negative"),
    ("HA4", "A", "A2a regret < generic gate (same features)",
     "sel", ("real", "A2a", "Gate", "all"), "negative"),
    ("HA5", "A", "A2a regret < conditional-risk selector (same features)",
     "sel", ("real", "A2a", "CondRisk", "all"), "negative"),
    ("HA6", "A", "A2a's gain over AlwaysG is larger with the real graph than with placebo",
     "selector_gain", ("A2a", "AlwaysG", "all"), "positive"),
    ("HA7", "A", "A2b (dual evidence) vs A2a regret [exploratory, two-sided]",
     "sel", ("real", "A2b", "A2a", "all"), "two-sided"),
]

ABLATIONS = [
    ("A1 vs A2a", "A2a", "A1"),
    ("A2a: with vs without structural diagnostics", "A2a", "A2a_noStruct"),
    ("A2a: with vs without D2 auxiliary", "A2a", "A2a_noAux"),
    ("A2a vs generic gate", "A2a", "Gate"),
    ("A2a vs conditional-risk", "A2a", "CondRisk"),
    ("A2b vs A2a", "A2b", "A2a"),
    ("A1 vs A2a_noStruct (model class only)", "A2a_noStruct", "A1"),
    ("A2a vs AlwaysN", "A2a", "AlwaysN"),
    ("A2a vs AlwaysG", "A2a", "AlwaysG"),
    ("A2a vs FixedEnsemble", "A2a", "FixedEnsemble"),
]


def load_seeds(run_dir: Path, cfg: dict) -> list[dict]:
    expected = list(cfg["experiment"]["seeds"])
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("configured seed list contains duplicates")
    expected_hash = config_hash(cfg)
    found = {}
    for p in sorted(run_dir.glob("seed_*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        seed = m.get("seed")
        if seed in found:
            raise ValueError(f"duplicate metrics for seed {seed} in {run_dir}")
        if m.get("config_hash") != expected_hash:
            raise ValueError(
                f"seed {seed} has config hash {m.get('config_hash')}, expected {expected_hash}"
            )
        found[seed] = m
    if not found:
        raise FileNotFoundError(f"no seed results in {run_dir}")
    found_set = set(found)
    if found_set != expected_set:
        missing = sorted(expected_set - found_set)
        extra = sorted(found_set - expected_set)
        raise ValueError(f"seed results do not match config; missing={missing}, extra={extra}")
    return [found[s] for s in expected]


def tidy(ms: list[dict]) -> pd.DataFrame:
    rows = []
    for m in ms:
        s = m["seed"]
        for slot, meths in m["selectors"].items():
            for meth, subs in meths.items():
                for sub, d in subs.items():
                    for k, v in d.items():
                        rows.append(dict(seed=s, table="selector", slot=slot, method=meth,
                                         subset=sub, metric=k, value=v))
        for sub, d in m["graph_value"].items():
            for k, v in d.items():
                rows.append(dict(seed=s, table="graph_value", slot="real", method="-",
                                 subset=sub, metric=k, value=v))
    return pd.DataFrame(rows)


def _sel(df, slot, method, subset, metric):
    q = df[(df.table == "selector") & (df.slot == slot) & (df.method == method)
           & (df.subset == subset) & (df.metric == metric)]
    return q.set_index("seed")["value"].astype(float).sort_index()


def _gv(df, subset, key):
    q = df[(df.table == "graph_value") & (df.subset == subset) & (df.metric == key)]
    return q.set_index("seed")["value"].astype(float).sort_index()


def hypothesis_diffs(df, kind, spec):
    if kind == "sel":
        slot, a, b, sub = spec
        return _sel(df, slot, a, sub, "regret") - _sel(df, slot, b, sub, "regret")
    if kind == "gv":
        return _gv(df, *spec)
    if kind == "gvd":
        s1, s2, key = spec
        return _gv(df, s1, key) - _gv(df, s2, key)
    if kind == "expert_slot":
        (sub,) = spec
        return (_sel(df, "placebo", "AlwaysG", sub, "cond_risk")
                - _sel(df, "real", "AlwaysG", sub, "cond_risk"))
    if kind == "selector_gain":
        method, baseline, sub = spec
        gain_real = (_sel(df, "real", baseline, sub, "cond_risk")
                     - _sel(df, "real", method, sub, "cond_risk"))
        gain_placebo = (_sel(df, "placebo", baseline, sub, "cond_risk")
                        - _sel(df, "placebo", method, sub, "cond_risk"))
        return gain_real - gain_placebo
    raise ValueError(kind)


def test_hypotheses(df, ecfg) -> pd.DataFrame:
    rows = []
    for hid, fam, desc, kind, spec, pred in HYPOTHESES:
        d = hypothesis_diffs(df, kind, spec)
        s = paired_summary(d.values, ecfg["ci_level"], ecfg["bootstrap_resamples"])
        rows.append(dict(id=hid, family=fam, hypothesis=desc, predicted_sign=pred, **s))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for fam in out.family.unique():
        m = (out.family == fam) & (out.predicted_sign != "two-sided")
        out.loc[m, "p_holm"] = holm(out.loc[m, "wilcoxon_p"].tolist())
    m = out.predicted_sign == "two-sided"
    out.loc[m, "p_holm"] = out.loc[m, "wilcoxon_p"]
    out["verdict"] = [verdict(r, r["p_holm"], r["predicted_sign"], ecfg["alpha"])
                      for _, r in out.iterrows()]
    return out


def ablation_table(df, ecfg) -> pd.DataFrame:
    rows = []
    for label, a, b in ABLATIONS:
        for slot in ("real", "placebo"):
            for sub in SUBSETS:
                d = _sel(df, slot, a, sub, "regret") - _sel(df, slot, b, sub, "regret")
                s = paired_summary(d.values, ecfg["ci_level"], ecfg["bootstrap_resamples"])
                rows.append(dict(comparison=label, A=a, B=b, slot=slot, subset=sub,
                                 metric="regret (A - B)", **s))
    return pd.DataFrame(rows)


def method_table(df, ecfg, slot="real") -> pd.DataFrame:
    rows = []
    metrics = ("regret", "cond_risk", "realised_mse", "graph_weight", "oracle_agreement",
               "benefit_rmse", "benefit_corr", "calib_slope", "calib_intercept", "calib_ece",
               "d2_rmse")
    for meth in METHODS:
        for sub in SUBSETS:
            for met in metrics:
                v = _sel(df, slot, meth, sub, met).dropna()
                if v.empty:
                    continue
                s = paired_summary(v.values, ecfg["ci_level"], 2000)
                rows.append(dict(slot=slot, method=meth, subset=sub, metric=met, mean=s["mean"],
                                 median=float(np.median(v.values)), ci_low=s["t_ci_low"], ci_high=s["t_ci_high"], n=s["n"]))
    return pd.DataFrame(rows)


def graph_value_table(df, ecfg) -> pd.DataFrame:
    rows = []
    keys = sorted(df[df.table == "graph_value"].metric.unique())
    for sub in SUBSETS:
        for k in keys:
            v = _gv(df, sub, k).dropna()
            s = paired_summary(v.values, ecfg["ci_level"], 2000)
            rows.append(dict(subset=sub, quantity=k, mean=s["mean"], ci_low=s["t_ci_low"],
                             ci_high=s["t_ci_high"], wilcoxon_p=s["wilcoxon_p"], n=s["n"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)


def plot_regret(mt: pd.DataFrame, path: Path, slot="real"):
    meths = [m for m in METHODS if m != "Oracle"]
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.27
    x = np.arange(len(meths))
    for j, reg in enumerate(REGIME_NAMES):
        sub = mt[(mt.metric == "regret") & (mt.subset == reg)].set_index("method").reindex(meths)
        ax.bar(x + (j - 1) * width, sub["mean"], width * 0.92, color=REGIME_COLORS[reg],
               label=reg, yerr=[sub["mean"] - sub.ci_low, sub.ci_high - sub["mean"]],
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    allr = mt[(mt.metric == "regret") & (mt.subset == "all")].set_index("method").reindex(meths)
    ax.scatter(x, allr["mean"], marker="D", s=28, color=INK, zorder=5, label="all regimes")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, meths, rotation=25, ha="right")
    ax.set_ylabel("Selector regret vs oracle (cond. risk)", fontsize=8, color=INK2)
    _style(ax, f"Locked-test regret by regime ({slot} graph slot), mean ± 95% CI over seeds")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_graph_value(df: pd.DataFrame, path: Path):
    qs = [("D1_realised_vs_mlp", "D1 vs matched MLP"),
          ("D1_realised_vs_cross_mlp", "D1 vs cross-asset MLP"),
          ("D1placebo_realised_vs_mlp", "D1 of placebo GCN vs MLP"),
          ("D2_realised", "D2 (placebo − real)")]
    fig, axes = plt.subplots(1, len(qs), figsize=(12, 3.4), sharey=True)
    for ax, (k, lab) in zip(axes, qs):
        for j, reg in enumerate(REGIME_NAMES):
            v = _gv(df, reg, k).values
            jitter = (np.arange(len(v)) - len(v) / 2) * 0.02
            ax.scatter(j + jitter, v, s=14, color=REGIME_COLORS[reg], alpha=0.75)
            ax.plot([j - 0.25, j + 0.25], [v.mean()] * 2, color=INK, lw=2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(3), REGIME_NAMES, rotation=20)
        _style(ax, lab)
    axes[0].set_ylabel("Realised MSE difference (test)", fontsize=8, color=INK2)
    fig.suptitle("Graph value by regime — dots are seeds, bars are means", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_selection(mt: pd.DataFrame, path: Path):
    meths = ["Oracle", "FixedEnsemble", "Gate", "CondRisk", "A1", "A2a", "A2a_noStruct",
             "A2a_noAux", "A2b"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, reg in zip(axes, REGIME_NAMES):
        sub = mt[(mt.metric == "graph_weight") & (mt.subset == reg)].set_index("method").reindex(meths)
        ax.bar(range(len(meths)), sub["mean"], 0.7, color=REGIME_COLORS[reg],
               yerr=[sub["mean"] - sub.ci_low, sub.ci_high - sub["mean"]],
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        ax.set_xticks(range(len(meths)), meths, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        _style(ax, f"{reg}")
    axes[0].set_ylabel("Mean weight on graph expert", fontsize=8, color=INK2)
    fig.suptitle("Graph selection frequency by regime (real slot, test)", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forest(ht: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, fam, lab in zip(axes, ("D", "A"), ("D: graph value (MSE units)", "A: selector comparisons")):
        sub = ht[ht.family == fam].iloc[::-1]
        y = np.arange(len(sub))
        ax.errorbar(sub["mean"], y, xerr=[sub["mean"] - sub.t_ci_low, sub.t_ci_high - sub["mean"]],
                    fmt="o", color="#2a78d6", ecolor=INK2, ms=6, capsize=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [f"{r.id} [{r.verdict}]" for r in sub.itertuples()], fontsize=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(lab, loc="left", fontsize=10)
    fig.suptitle("Pre-specified hypotheses: mean paired difference, 95% t-CI over seeds",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _test_mask(run_dir, seed, cfg):
    he = np.load(run_dir / f"seed_{seed:02d}" / "hidden_eval.npz")
    t = he["decision_time_0based"]
    return he, t >= cfg["splits"]["test"][0] - 1


def plot_timeline(run_dir: Path, cfg, path: Path, seed: int):
    he, m = _test_mask(run_dir, seed, cfg)
    so = np.load(run_dir / f"seed_{seed:02d}" / "selector_outputs.npz")
    t = he["decision_time_0based"][m] + 1
    reg = he["regime"][m]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for r, n in enumerate(REGIME_NAMES):
        ax.fill_between(t, -1.2, 1.2, where=reg == r, color=REGIME_COLORS[n], alpha=0.12,
                        step="mid", label=f"regime: {n}", lw=0)
    ax.plot(t, he["benefit_star"][m], color=INK, lw=1.2, label="true conditional benefit B*")
    ax.plot(t, so["real/A1/benefit"][m], color="#6250d6", lw=1.2, label="A1 estimate")
    ax.plot(t, so["real/A2a/benefit"][m], color="#e34948", lw=1.2, label="A2a estimate")
    ax.axhline(0, color=INK2, lw=0.8)
    lim = np.nanmax(np.abs(he["benefit_star"][m])) * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("time step (1-based)", fontsize=8, color=INK2)
    ax.set_ylabel("benefit of graph expert", fontsize=8, color=INK2)
    _style(ax, f"Seed {seed}: estimated vs true graph benefit on the locked test")
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(run_dir: Path, cfg, seeds, path: Path, bins=10):
    meths = [("A1", "#6250d6"), ("A2a", "#e34948"), ("A2a_noStruct", "#2a78d6"),
             ("CondRisk", "#eda100")]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    allv = []
    for meth, col in meths:
        P, Y = [], []
        for s in seeds:
            he, m = _test_mask(run_dir, s, cfg)
            so = np.load(run_dir / f"seed_{s:02d}" / "selector_outputs.npz")
            P.append(so[f"real/{meth}/benefit"][m])
            Y.append(he["benefit_star"][m])
        P, Y = np.concatenate(P), np.concatenate(Y)
        edges = np.quantile(P, np.linspace(0, 1, bins + 1))
        w = np.clip(np.searchsorted(edges, P, side="right") - 1, 0, bins - 1)
        px = [P[w == b].mean() for b in range(bins)]
        py = [Y[w == b].mean() for b in range(bins)]
        ax.plot(px, py, "-o", color=col, ms=4, lw=1.5, label=meth)
        allv += px + py
    lo, hi = min(allv), max(allv)
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--", label="perfect calibration")
    ax.set_xlabel("predicted benefit (decile mean)", fontsize=8, color=INK2)
    ax.set_ylabel("true conditional benefit B* (decile mean)", fontsize=8, color=INK2)
    _style(ax, "Benefit calibration, pooled test, all seeds")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_report(run_dir: Path, cfg, ht, mt, gvt, abl, ms):
    seeds = [m["seed"] for m in ms]
    L = []
    L.append("# D + A synthetic pilot — results report\n")
    L.append(f"Run: `{run_dir.name}` · config hash `{ms[0]['config_hash']}` · seeds {seeds}\n")
    L.append("All numbers are locked-test (time steps 2001–3000) and aggregated over seeds; "
             "intervals are 95% t-intervals over seeds (n = NSEEDS). Hypothesis verdicts use a "
             "two-sided Wilcoxon signed-rank test on per-seed paired differences, Holm-adjusted "
             "within each family (D, A), plus the requirement that the t-interval excludes zero "
             "in the stated direction.\n".replace("NSEEDS", str(len(seeds))))
    L.append("## Hypotheses\n")
    L.append("| ID | Hypothesis | Mean diff | 95% CI | Wilcoxon p (Holm) | +/− seeds | Verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for r in ht.itertuples():
        L.append(f"| {r.id} | {r.hypothesis} | {_fmt(r.mean, 4)} | [{_fmt(r.t_ci_low, 4)}, "
                 f"{_fmt(r.t_ci_high, 4)}] | {_fmt(r.p_holm, 4)} | {r.n_positive}/{r.n_negative} | "
                 f"**{r.verdict}** |")
    L.append("\nSign convention: D rows are the quantity itself (or a regime difference); A rows "
             "are regret(A) − regret(B), so negative favours the first-named method. HD7 isolates "
             "the raw real-vs-placebo expert-quality gap. HA6 is a difference-in-differences: "
             "A2a's conditional-risk gain over AlwaysG in the real slot minus that gain in the "
             "placebo slot.\n")

    L.append("## Selector regret by regime (real graph slot)\n")
    L.append("| Method | all | informative | no_signal | corrupted | graph weight (all) | oracle agreement |")
    L.append("|---|---|---|---|---|---|---|")
    for meth in METHODS:
        cells = []
        for sub in SUBSETS:
            q = mt[(mt.method == meth) & (mt.subset == sub) & (mt.metric == "regret")]
            cells.append(f"{_fmt(q['mean'].iloc[0], 4)} ± {_fmt((q.ci_high - q['mean']).iloc[0], 4)}"
                         if len(q) else "–")
        gw = mt[(mt.method == meth) & (mt.subset == "all") & (mt.metric == "graph_weight")]
        oa = mt[(mt.method == meth) & (mt.subset == "all") & (mt.metric == "oracle_agreement")]
        L.append(f"| {meth} | " + " | ".join(cells) +
                 f" | {_fmt(gw['mean'].iloc[0], 2) if len(gw) else '–'} | "
                 f"{_fmt(oa['mean'].iloc[0], 2) if len(oa) else '–'} |")

    L.append("\n## Graph value (D1, D2) by regime\n")
    L.append("| Quantity | all | informative | no_signal | corrupted |")
    L.append("|---|---|---|---|---|")
    for k in ("D1_realised_vs_mlp", "D1_oracle_vs_mlp", "D1_realised_vs_cross_mlp",
              "D1_realised_vs_ridge", "D1placebo_realised_vs_mlp", "D2_realised", "D2_oracle"):
        cells = []
        for sub in SUBSETS:
            q = gvt[(gvt.subset == sub) & (gvt.quantity == k)]
            cells.append(f"{_fmt(q['mean'].iloc[0], 4)} [{_fmt(q.ci_low.iloc[0], 3)}, {_fmt(q.ci_high.iloc[0], 3)}]")
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    L.append("\n## Graph-value estimation error and calibration (real slot, all test steps)\n")
    L.append("| Method | benefit RMSE vs B* | corr with B* | calib. slope (median) | calib. slope (mean) | calib. intercept (median) | binned calib. error | D2 RMSE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for meth in ("CondRisk", "A1", "A2a", "A2a_noStruct", "A2a_noAux"):
        def g(met, stat="mean"):
            q = mt[(mt.method == meth) & (mt.subset == "all") & (mt.metric == met)]
            return _fmt(q[stat].iloc[0], 3) if len(q) else "–"
        L.append(f"| {meth} | {g('benefit_rmse')} | {g('benefit_corr')} | "
                 f"{g('calib_slope', 'median')} | {g('calib_slope')} | "
                 f"{g('calib_intercept', 'median')} | {g('calib_ece')} | {g('d2_rmse')} |")
    L.append("\nCalibration slope = OLS slope of true benefit B* on the calibrated prediction "
             "(ideal 1). The seed mean is unstable: in seeds where a selector's test-period "
             "predictions are nearly constant the slope explodes (per-seed values in "
             "`per_seed_results.csv`), so the median is the robust summary.")

    L.append("\n## Ablation summary (regret A − B, real slot)\n")
    L.append("| Comparison | all | informative | no_signal | corrupted |")
    L.append("|---|---|---|---|---|")
    for label in dict.fromkeys(abl.comparison):
        cells = []
        for sub in SUBSETS:
            q = abl[(abl.comparison == label) & (abl.slot == "real") & (abl.subset == sub)].iloc[0]
            cells.append(f"{_fmt(q['mean'], 4)} [{_fmt(q.t_ci_low, 4)}, {_fmt(q.t_ci_high, 4)}]")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L.append("\n## Plots\n")
    for p in ("regret_by_regime.png", "regret_by_regime_placebo.png", "graph_value_by_regime.png",
              "selection_frequency.png", "hypotheses_forest.png", "calibration.png",
              f"timeline_seed{seeds[0]:02d}.png"):
        L.append(f"- `plots/{p}`")
    L.append("\n## Caveats\n")
    L.append("- This is a synthetic pilot. The data-generating process was designed so that "
             "corruption of the observed adjacency is, in principle, visible to structural "
             "diagnostics, while the no-signal regime is structurally indistinguishable from the "
             "informative one. Any advantage of structural conditioning is therefore conditional "
             "on that design and is not evidence about real markets.\n"
             "- Experts are frozen after expert-val; no re-training through time.\n"
             "- n = 10 seeds; Wilcoxon's smallest attainable two-sided p is ≈0.002, so Holm-"
             "adjusted significance requires near-unanimous sign agreement across seeds.\n"
             "- No claim of novelty or of guaranteed robustness is made; results describe this "
             "benchmark only.\n")
    (run_dir / "report.md").write_text("\n".join(L))


def aggregate(run_dir: str | Path, cfg: dict) -> dict:
    run_dir = Path(run_dir)
    ecfg = cfg["evaluation"]
    ms = load_seeds(run_dir, cfg)
    df = tidy(ms)
    df.to_csv(run_dir / "per_seed_results.csv", index=False)
    ht = test_hypotheses(df, ecfg)
    ht.to_csv(run_dir / "hypotheses.csv", index=False)
    mt = method_table(df, ecfg, "real")
    mtp = method_table(df, ecfg, "placebo")
    pd.concat([mt, mtp]).to_csv(run_dir / "method_summary.csv", index=False)
    gvt = graph_value_table(df, ecfg)
    gvt.to_csv(run_dir / "graph_value_summary.csv", index=False)
    abl = ablation_table(df, ecfg)
    abl.to_csv(run_dir / "ablation_summary.csv", index=False)

    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    seeds = [m["seed"] for m in ms]
    plot_regret(mt, plots / "regret_by_regime.png")
    plot_regret(mtp, plots / "regret_by_regime_placebo.png", slot="placebo")
    plot_graph_value(df, plots / "graph_value_by_regime.png")
    plot_selection(mt, plots / "selection_frequency.png")
    plot_forest(ht, plots / "hypotheses_forest.png")
    plot_calibration(run_dir, cfg, seeds, plots / "calibration.png", ecfg["calibration_bins"])
    plot_timeline(run_dir, cfg, plots / f"timeline_seed{seeds[0]:02d}.png", seeds[0])
    write_report(run_dir, cfg, ht, mt, gvt, abl, ms)

    summary = {
        "run": run_dir.name, "seeds": seeds, "config_hash": ms[0]["config_hash"],
        "hypotheses": json.loads(ht.to_json(orient="records")),
        "methods_real": json.loads(mt.to_json(orient="records")),
        "graph_value": json.loads(gvt.to_json(orient="records")),
    }
    with open(run_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary
