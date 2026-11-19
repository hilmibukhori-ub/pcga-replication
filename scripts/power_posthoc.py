"""Post hoc sensitivity (review request): what effects could the confirmatory tests have detected?
Not 'observed power'. For each D1 test: the one-sided 95% upper confidence bound, and the gain detectable
with 80% power by a one-sided paired t-test at alpha = 0.05 (unadjusted) and at the Holm first-step level
alpha/m, using the observed SD of the per-seed differences. For each edge rank test: the shift of the real
graph, in placebo SDs and in IRR units, that gives 80% power, assuming real and placebo outcomes share the
placebo spread (normal; simulation)."""
import json, pathlib, numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[1]
rsr = json.load(open(ROOT / "outputs/rsr_replica_v1/results.json"))
rt = json.load(open(ROOT / "outputs/rtgcn_v1/results.json"))
ext = json.load(open(ROOT / "outputs/rtgcn_ext_v1/results.json"))
sens = json.load(open(ROOT / "outputs/rtgcn_ext_v1/sensitivity_excl_day89.json"))

def d1(name, est, sd, n, m):
    se = sd / np.sqrt(n); df = n - 1
    mde = lambda a: float((stats.t.ppf(1 - a, df) + stats.t.ppf(0.8, df)) * se)
    return dict(test=name, estimate=est, sd_diff=sd, n=n, upper95_one_sided=float(est + stats.t.ppf(0.95, df) * se),
                mde80_alpha05=mde(0.05), mde80_holm_first_step=mde(0.05 / m), holm_family_size=m)

def sd_from_ci(ci, n):
    return (ci[1] - ci[0]) / 2 * np.sqrt(n) / stats.t.ppf(0.975, n - 1)

rows = []
for t in rsr["tests"]:
    if t["quantity"] == "D1_IRR":
        rows.append(d1(f"RSR {t['market']} D1", t["estimate"], t["sd_seed_diff"], t["n_seeds"], 4))
hb = [c for c in rt["confirmatory"] if c["id"].startswith("H-B")][0]
rows.append(d1("RT-GCN NASDAQ-U D1", hb["estimate"], sd_from_ci(hb["ci95"], hb["n"]), hb["n"], 2))
for c in ext["confirmatory"]:
    if c["id"].startswith("H-B"):
        cell = c["id"][4:]
        rows.append(d1(f"RT-GCN {cell} D1", c["estimate"], sd_from_ci(c["ci95"], c["n"]), c["n"], 6))
        s = sens[cell]["H-B_excl89"]
        rows.append(d1(f"RT-GCN {cell} D1 (excl. day 89, exploratory)", s["estimate"], sd_from_ci(s["ci95"], s["n"]), s["n"], 6))

rng = np.random.default_rng(20261111)
def edge_power(K, shift, reps=40000, alpha=0.05):
    pl = rng.standard_normal((reps, K)); real = rng.standard_normal(reps) + shift
    p = (1 + (pl >= real[:, None]).sum(1)) / (K + 1)
    return float((p <= alpha).mean())
def edge_shift80(K):
    lo, hi = 0.0, 6.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if edge_power(K, mid) < 0.8 else (lo, mid)
    return hi
s19, s99 = edge_shift80(19), edge_shift80(99)
edge = [dict(test="RSR NASDAQ edge", K=99, placebo_sd=rsr["descriptive"]["NASDAQ"]["placebo_irr"]["sd"]),
        dict(test="RSR NYSE edge", K=99, placebo_sd=rsr["descriptive"]["NYSE"]["placebo_irr"]["sd"]),
        dict(test="RT-GCN NASDAQ-U edge", K=19, placebo_sd=rt["H-C_edge_IRR1"]["placebo_sd"])]
for cell in ("NYSE_U", "NASDAQ_T", "NYSE_T"):
    edge.append(dict(test=f"RT-GCN {cell} edge", K=19, placebo_sd=ext["cells"][cell]["H-C_edge_IRR1"]["placebo_sd"]))
for e in edge:
    e["shift80_in_placebo_sd"] = s19 if e["K"] == 19 else s99
    e["shift80_irr"] = e["shift80_in_placebo_sd"] * e["placebo_sd"]
out = dict(d1=rows, edge=edge, note=__doc__)
json.dump(out, open(ROOT / "outputs/power_posthoc.json", "w"), indent=1)
for r in rows:
    print(f"{r['test']:<48} est {r['estimate']:+.3f}  upper95 {r['upper95_one_sided']:+.3f}  MDE80 {r['mde80_alpha05']:.3f} (Holm-1st {r['mde80_holm_first_step']:.3f})")
for e in edge:
    print(f"{e['test']:<30} K={e['K']}  80% power needs shift {e['shift80_in_placebo_sd']:.2f} placebo SD = {e['shift80_irr']:.2f} IRR")
