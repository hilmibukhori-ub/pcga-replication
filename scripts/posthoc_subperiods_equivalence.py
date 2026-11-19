"""Post hoc, descriptive (review request). No confirmatory verdict changes.
(1) Temporal robustness without retraining: the test period of each study is cut into consecutive blocks,
    and D1 (paired over seeds) and the edge rank are recomputed per block from the stored daily top-1
    returns of the already selected epoch. This re-evaluates the same trained models on sub-windows; it is
    not a rolling re-training design.
(2) Equivalence and inferiority-to-published:
    TOST for |D1| < delta, where delta = 0.09 IRR, the "significant improvement" threshold stated in the RSR
    paper. One-sided test of H0: D1 >= published gain (that a matched gain as large as the published gain is
    present)."""
import json, pathlib, numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[1]
O_RSR, O_RT, O_EXT = ROOT / "outputs/rsr_replica_v1", ROOT / "outputs/rtgcn_v1", ROOT / "outputs/rtgcn_ext_v1"

def rsr_daily(mkt, g, s): return np.load(O_RSR / f"{mkt}_wikidata_{g}_s{s}.npz")["test_top1"].astype(float)
def rt_daily(path):
    z = np.load(path); e = int(np.argmin(z["val_loss"])); return z["test_top1"][e].astype(float)
def rt_path(cell, g, s):
    return (O_RT / f"NASDAQ_val_paper_{g}_s{s}.npz") if cell == "NASDAQ_U" else (O_EXT / f"{cell}_val_paper_{g}_s{s}.npz")

def blocks(n, k): return np.array_split(np.arange(n), k)
def pt(d):
    d = np.asarray(d); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, d.mean(), se)
    return dict(est=float(d.mean()), ci95=[float(lo), float(hi)], p_one_sided=float(stats.t.sf(d.mean() / se, n - 1)), n_pos=int((d > 0).sum()), n=n)

studies = {}
# RSR: seeds 0-19 for D1, perm000-098 at seed 0
for mkt in ("NASDAQ", "NYSE"):
    R = np.array([rsr_daily(mkt, "real", s) for s in range(20)]); N = np.array([rsr_daily(mkt, "none", s) for s in range(20)])
    P = np.array([rsr_daily(mkt, f"perm{k:03d}", 0) for k in range(99)])
    studies[f"RSR_{mkt}"] = (R, N, P)
cells = {"NASDAQ_U": 20, "NYSE_U": 20, "NASDAQ_T": 10, "NYSE_T": 10}
for cell, S in cells.items():
    R = np.array([rt_daily(rt_path(cell, "real", s)) for s in range(S)]); N = np.array([rt_daily(rt_path(cell, "none", s)) for s in range(S)])
    P = np.array([rt_daily(rt_path(cell, f"perm{k:03d}", 0)) for k in range(19)])
    studies[f"RTGCN_{cell}"] = (R, N, P)

res = {"subperiods": {}, "equivalence": {}}
for name, (R, N, P) in studies.items():
    full = pt(R.sum(1) - N.sum(1))
    rows = []
    for j, idx in enumerate(blocks(R.shape[1], 4)):
        d = R[:, idx].sum(1) - N[:, idx].sum(1)
        Tr, Tp = R[0, idx].sum(), P[:, idx].sum(1)
        rows.append(dict(block=j + 1, days=[int(idx[0]), int(idx[-1])], D1=pt(d),
                         edge_rank=int(1 + (Tp >= Tr).sum()), K=len(Tp), edge_p=float((1 + (Tp >= Tr).sum()) / (len(Tp) + 1))))
    res["subperiods"][name] = dict(full_D1=full, blocks=rows)
    print(f"\n{name}: full D1 {full['est']:+.3f} {np.round(full['ci95'],3).tolist()}")
    for r in rows:
        print(f"  block {r['block']} days {r['days']}: D1 {r['D1']['est']:+.3f} {np.round(r['D1']['ci95'],3).tolist()} "
              f"p1s {r['D1']['p_one_sided']:.3f} ({r['D1']['n_pos']}/{r['D1']['n']})  edge rank {r['edge_rank']}/{r['K']+1}")

PUB = {"RSR_NASDAQ": 0.51, "RSR_NYSE": 0.23, "RTGCN_NASDAQ_U": 0.10, "RTGCN_NYSE_U": 0.35, "RTGCN_NASDAQ_T": 0.90, "RTGCN_NYSE_T": 0.62}
DELTA = 0.09
for name, (R, N, P) in studies.items():
    d = R.sum(1) - N.sum(1); n = len(d); m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    p_low = stats.t.sf((m + DELTA) / se, n - 1); p_up = stats.t.cdf((m - DELTA) / se, n - 1)
    lo90, hi90 = stats.t.interval(0.90, n - 1, m, se)
    p_inf = stats.t.cdf((m - PUB[name]) / se, n - 1)
    res["equivalence"][name] = dict(delta=DELTA, ci90=[float(lo90), float(hi90)], p_tost=float(max(p_low, p_up)),
                                    equivalent=bool(max(p_low, p_up) <= 0.05), published_gain=PUB[name],
                                    p_gain_below_published=float(p_inf))
    e = res["equivalence"][name]
    print(f"{name}: 90% CI {np.round(e['ci90'],3).tolist()}  TOST(±0.09) p {e['p_tost']:.3f} equivalent={e['equivalent']}  "
          f"H0 D1>=published {PUB[name]}: p {e['p_gain_below_published']:.4f}")
res["note"] = __doc__
json.dump(res, open(ROOT / "outputs/posthoc_subperiods_equivalence.json", "w"), indent=1)
