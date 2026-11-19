"""Analysis for the RT-GCN re-test (PREREG_RTGCN.md). Written before the confirm run.

Every run stores, for every epoch, the val/test mean loss and the daily top-1/5/10 returns and reciprocal
ranks. The selection rules are applied here:
  e_test = argmin test loss   (the published rule in processor/recognition.py)
  e_val  = argmin val loss    (a proper validation split)
  e_last = final epoch
Metrics: IRR-k = sum over test days of the mean realised return of the top-k predicted stocks. "all" uses
every test day. "pub" deletes test day 89, as the released code does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rtgcn_v1"
DELETE_DAY = 89
PAPER_U_NASDAQ = {"mrr": 0.055, "irr1": 0.45, "irr5": 0.71, "irr10": 0.82}      # Table IV, RT-GCN (U)
PAPER_RANK_LSTM_NASDAQ = {"mrr": 0.041, "irr1": 0.35, "irr5": 0.42, "irr10": 0.82}


def load(arm, cfgname, graph, seeds, out, market="NASDAQ"):
    fs = [out / f"{market}_{arm}_{cfgname}_{graph}_s{s}.npz" for s in seeds]
    return [dict(np.load(f)) for f in fs] if all(f.exists() for f in fs) else None


def sel(run, rule):
    if rule == "test":
        return int(np.argmin(run["test_loss"]))
    if rule == "val":
        return int(np.argmin(run["val_loss"]))
    if rule == "last":
        return len(run["test_loss"]) - 1
    raise ValueError(rule)


def metric(run, e, name="irr1", days="all"):
    keep = np.ones(run["test_top1"].shape[1], bool)
    if days == "pub":
        keep[DELETE_DAY] = False
    if name.startswith("irr"):
        return float(run[f"test_top{name[3:]}"][e][keep].sum())
    if name == "mrr":
        return float(run["test_rr"][e][keep].mean())
    raise ValueError(name)


def paired(a, b):
    """One-sided paired t (H1: mean(a - b) > 0), plus estimate, 95% CI, and count of a > b."""
    d = np.asarray(a) - np.asarray(b)
    t = stats.ttest_rel(a, b, alternative="greater")
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    return {"estimate": float(d.mean()), "ci95": [float(lo), float(hi)], "p": float(t.pvalue),
            "n": int(len(d)), "n_pos": int((d > 0).sum())}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def summarise(cfg, out=OUT, arms=("val", "pub")):
    """cfg: configs/rtgcn_replica.yaml. arms=("dev", "devpub") for the dev rehearsal."""
    VAL, PUB = arms
    S = range(cfg["seeds"])
    K = cfg["placebos"]
    real, none = load(VAL, "paper", "real", S, out), load(VAL, "paper", "none", S, out)
    res = {"complete": {}}
    tests = []
    if real is not None:
        # H-A: inflation from test-set epoch selection (real graph, val arm)
        a = [metric(r, sel(r, "test")) for r in real]
        b = [metric(r, sel(r, "val")) for r in real]
        tests.append({"id": "H-A_inflation_IRR1", **paired(a, b),
                      "mean_test_selected": float(np.mean(a)), "mean_val_selected": float(np.mean(b))})
    if real is not None and none is not None:
        # H-B: D1 (relations vs none), validation-selected epochs
        a = [metric(r, sel(r, "val")) for r in real]
        b = [metric(n, sel(n, "val")) for n in none]
        tests.append({"id": "H-B_D1_IRR1", **paired(a, b), "mean_real": float(np.mean(a)), "mean_none": float(np.mean(b))})
    if tests:
        for t, adj in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = adj
            t["verdict"] = "supported" if adj <= cfg["alpha"] else "not detected"
    res["confirmatory"] = tests
    # H-C (separate, exact rank test, K placebos at seed 0, validation-selected)
    pl = [load(VAL, "paper", f"perm{k:03d}", [0], out) for k in range(K)]
    if real is not None and all(p is not None for p in pl):
        Tr = metric(real[0], sel(real[0], "val"))
        Tp = np.array([metric(p[0], sel(p[0], "val")) for p in pl])
        res["H-C_edge_IRR1"] = {"T_real": Tr, "placebo_mean": float(Tp.mean()), "placebo_sd": float(Tp.std(ddof=1)),
                                "n_at_least_real": int((Tp >= Tr).sum()), "p_rank": float((1 + (Tp >= Tr).sum()) / (K + 1))}
    # descriptive
    desc = {}
    for nm in ("irr1", "irr5", "irr10", "mrr"):
        for rule in ("test", "val", "last"):
            if real is not None:
                desc[f"real_{rule}_{nm}"] = [metric(r, sel(r, rule), nm) for r in real]
            if none is not None and rule != "test":
                desc[f"none_{rule}_{nm}"] = [metric(n, sel(n, rule), nm) for n in none]
    if real is not None:
        desc["epochs_real"] = {r: [sel(x, r) for x in real] for r in ("test", "val")}
    ind = load(VAL, "paper", "ind", range(cfg["seeds_ind"]), out)
    if ind is not None and none is not None:
        desc["D1_industry_only_IRR1"] = paired([metric(x, sel(x, "val")) for x in ind],
                                               [metric(n, sel(n, "val")) for n in none[:len(ind)]])
    pub = load(PUB, "paper", "real", range(cfg["seeds_pub"]), out)
    if pub is not None:
        desc["R0_published_protocol"] = {nm: {"mean": float(np.mean([metric(r, sel(r, "test"), nm, "pub") for r in pub])),
                                              "sd": float(np.std([metric(r, sel(r, "test"), nm, "pub") for r in pub], ddof=1)),
                                              "paper": PAPER_U_NASDAQ[nm]} for nm in ("irr1", "irr5", "irr10", "mrr")}
    S_rel = range(cfg["seeds_released"])
    rr_, rn_ = load(VAL, "released", "real", S_rel, out), load(VAL, "released", "none", S_rel, out)
    if rr_ is not None and rn_ is not None:
        desc["released_cfg_D1_IRR1"] = paired([metric(x, sel(x, "val")) for x in rr_], [metric(x, sel(x, "val")) for x in rn_])
        desc["released_cfg_inflation_IRR1"] = paired([metric(x, sel(x, "test")) for x in rr_], [metric(x, sel(x, "val")) for x in rr_])
    if real is not None:
        E = np.stack([r["test_top1"].sum(1) for r in real])          # (seeds, epochs) IRR-1 per epoch
        desc["variance_IRR1"] = {"within_run_sd_over_epochs_mean": float(E.std(1, ddof=1).mean()),
                                 "between_seed_sd_at_val_epoch": float(np.std([metric(r, sel(r, "val")) for r in real], ddof=1)),
                                 "corr_testloss_IRR1_mean": float(np.mean([np.corrcoef(r["test_loss"], r["test_top1"].sum(1))[0, 1] for r in real])),
                                 "oracle_best_epoch_IRR1_mean": float(E.max(1).mean())}
    res["descriptive"] = desc
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    lines = ["# RT-GCN re-test — results", ""]
    for t in tests:
        lines.append(f"- **{t['id']}**: estimate {t['estimate']:+.3f} [95% CI {t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}], "
                     f"p {t['p']:.4f}, Holm {t['p_holm']:.4f} → {t['verdict']} ({t['n_pos']}/{t['n']} seeds positive)")
    if "H-C_edge_IRR1" in res:
        h = res["H-C_edge_IRR1"]
        lines.append(f"- **H-C edge** (exact rank, K={K}): real {h['T_real']:+.3f} vs placebo mean {h['placebo_mean']:+.3f}; "
                     f"{h['n_at_least_real']}/{K} placebos ≥ real; p = {h['p_rank']:.3f}")
    if "R0_published_protocol" in desc:
        lines.append("- **R0 (published protocol, test-selected, day 89 deleted)**: " + ", ".join(
            f"{k} {v['mean']:.3f}±{v['sd']:.3f} (paper {v['paper']})" for k, v in desc["R0_published_protocol"].items()))
    for k in ("released_cfg_D1_IRR1", "released_cfg_inflation_IRR1", "D1_industry_only_IRR1"):
        if k in desc:
            v = desc[k]
            lines.append(f"- {k} (descriptive): {v['estimate']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], p {v['p']:.4f}, {v['n_pos']}/{v['n']}")
    if "variance_IRR1" in desc:
        lines.append("- variance_IRR1: " + ", ".join(f"{k} {v:.3f}" for k, v in desc["variance_IRR1"].items()))
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return res
