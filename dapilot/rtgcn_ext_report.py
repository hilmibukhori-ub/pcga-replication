"""Analysis for the RT-GCN extension (PREREG_RTGCN_EXT.md). Written before any extension confirm run.

Reuses the frozen main-study helpers (dapilot/rtgcn_report.py): epoch selection `sel`, metric `metric`,
one-sided paired t `paired`, `holm`. Per cell (NYSE_U, NASDAQ_T, NYSE_T):
  H-A  IRR-1(e_test) - IRR-1(e_val), real graph, val arm, paired by seed
  H-B  IRR-1(real, e_val) - IRR-1(none, e_val), val arm, paired by seed
  -> one Holm family over the 2 x 3 = 6 tests.
  H-C  exact placebo rank at seed 0 (K placebos), per cell, unadjusted, reported apart.
  R0   pub arm (test-selected, day 89 deleted) vs Table IV, descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .rtgcn_report import holm, metric, paired, sel

ROOT = Path(__file__).resolve().parents[1]
PAPER = {  # Table IV (mean of 15 runs)
    "NYSE_U": {"mrr": 0.040, "irr1": 0.65, "irr5": 0.88, "irr10": 0.96},
    "NASDAQ_T": {"mrr": 0.061, "irr1": 1.25, "irr5": 0.97, "irr10": 1.03},
    "NYSE_T": {"mrr": 0.056, "irr1": 0.92, "irr5": 1.10, "irr10": 1.13},
}
PAPER_RANK_LSTM = {"NASDAQ": 0.35, "NYSE": 0.30}   # IRR-1, Table IV


def load(out, cell, arm, graph, seeds):
    market, variant = cell.split("_")
    fs = [out / f"{market}_{variant}_{arm}_paper_{graph}_s{s}.npz" for s in seeds]
    return [dict(np.load(f)) for f in fs] if all(f.exists() for f in fs) else None


def summarise(cfg, out=None, arms=("val", "pub")):
    out = Path(out) if out else ROOT / cfg["output_dir"]
    VAL, PUB = arms
    tests, res = [], {"cells": {}}
    for cell in cfg["cells"]:
        S, K, P = range(cfg["seeds"][cell]), cfg["placebos"][cell], range(cfg["seeds_pub"][cell])
        real, none = load(out, cell, VAL, "real", S), load(out, cell, VAL, "none", S)
        c = {"complete": {"real": real is not None, "none": none is not None}}
        if real is not None:
            a = [metric(r, sel(r, "test")) for r in real]
            b = [metric(r, sel(r, "val")) for r in real]
            tests.append({"id": f"H-A_{cell}", **paired(a, b), "mean_test_selected": float(np.mean(a)),
                          "mean_val_selected": float(np.mean(b))})
        if real is not None and none is not None:
            a = [metric(r, sel(r, "val")) for r in real]
            b = [metric(n, sel(n, "val")) for n in none]
            tests.append({"id": f"H-B_{cell}", **paired(a, b), "mean_real": float(np.mean(a)), "mean_none": float(np.mean(b))})
        pl = [load(out, cell, VAL, f"perm{k:03d}", [0]) for k in range(K)]
        if real is not None and all(p is not None for p in pl):
            Tr = metric(real[0], sel(real[0], "val"))
            Tp = np.array([metric(p[0], sel(p[0], "val")) for p in pl])
            c["H-C_edge_IRR1"] = {"T_real": Tr, "placebo_mean": float(Tp.mean()), "placebo_sd": float(Tp.std(ddof=1)),
                                  "n_at_least_real": int((Tp >= Tr).sum()), "p_rank": float((1 + (Tp >= Tr).sum()) / (K + 1))}
        desc = {}
        for nm in ("irr1", "irr5", "irr10", "mrr"):
            for rule in ("test", "val", "last"):
                if real is not None:
                    desc[f"real_{rule}_{nm}"] = [metric(r, sel(r, rule), nm) for r in real]
                if none is not None and rule != "test":
                    desc[f"none_{rule}_{nm}"] = [metric(n, sel(n, rule), nm) for n in none]
        pub = load(out, cell, PUB, "real", P)
        if pub is not None:
            r0 = {}
            for nm in ("irr1", "irr5", "irr10", "mrr"):
                v = np.array([metric(r, sel(r, "test"), nm, "pub") for r in pub])
                lo, hi = stats.t.interval(0.95, len(v) - 1, v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
                r0[nm] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci95": [float(lo), float(hi)],
                          "paper": PAPER[cell][nm], "reproduction_gap": bool(not (lo <= PAPER[cell][nm] <= hi))}
            desc["R0_published_protocol"] = r0
        if real is not None:
            E = np.stack([r["test_top1"].sum(1) for r in real])
            desc["variance_IRR1"] = {"within_run_sd_over_epochs_mean": float(E.std(1, ddof=1).mean()),
                                     "between_seed_sd_at_val_epoch": float(np.std([metric(r, sel(r, "val")) for r in real], ddof=1)),
                                     "corr_testloss_IRR1_mean": float(np.mean([np.corrcoef(r["test_loss"], r["test_top1"].sum(1))[0, 1] for r in real])),
                                     "oracle_best_epoch_IRR1_mean": float(E.max(1).mean())}
        c["descriptive"] = desc
        res["cells"][cell] = c
    if tests:
        for t, adj in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = adj
            t["verdict"] = "supported" if adj <= cfg["alpha"] else "not detected"
    res["confirmatory"] = tests
    res["family_complete"] = len(tests) == 2 * len(cfg["cells"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    lines = ["# RT-GCN extension — results", "",
             f"Holm family complete: {res['family_complete']} ({len(tests)} of {2 * len(cfg['cells'])} tests).", ""]
    for t in tests:
        lines.append(f"- **{t['id']}**: {t['estimate']:+.3f} [95% CI {t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}], "
                     f"p {t['p']:.4f}, Holm {t['p_holm']:.4f} → {t['verdict']} ({t['n_pos']}/{t['n']} seeds positive)")
    for cell, c in res["cells"].items():
        if "H-C_edge_IRR1" in c:
            h = c["H-C_edge_IRR1"]
            K = cfg["placebos"][cell]
            lines.append(f"- **H-C {cell}** (exact rank, K={K}, unadjusted): real {h['T_real']:+.3f} vs placebo mean "
                         f"{h['placebo_mean']:+.3f}; {h['n_at_least_real']}/{K} ≥ real; p = {h['p_rank']:.3f}")
        r0 = c["descriptive"].get("R0_published_protocol")
        if r0:
            lines.append(f"- R0 {cell}: " + ", ".join(f"{k} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] "
                                                     f"(paper {v['paper']}{', GAP' if v['reproduction_gap'] else ''})"
                                                     for k, v in r0.items()))
        if "variance_IRR1" in c["descriptive"]:
            lines.append(f"- variance {cell}: " + ", ".join(f"{k} {v:.3f}" for k, v in c["descriptive"]["variance_IRR1"].items()))
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return res
