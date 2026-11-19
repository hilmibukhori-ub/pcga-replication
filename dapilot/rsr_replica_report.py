"""Analysis for the RSR replication (PREREG_RSR_REPLICATION.md §6-§8). Fixed before the test run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rsr_pipeline import bootstrap_p_one_sided
from . import rsr_replica_infer as infer
from .stats import holm

ROOT = Path(__file__).resolve().parents[1]
PAPER = {  # Feng et al. 2019, TOIS, Table 7 (Wiki) and Table 6 (industry), mean ± sd over 5 runs
    "NASDAQ": {"RSR_I_wiki": {"mse": 3.79e-4, "mrr": 4.09e-2, "irr": 1.19, "irr_sd": 0.55},
               "RSR_I_industry": {"mse": 3.80e-4, "mrr": 3.17e-2, "irr": 0.23, "irr_sd": 0.27},
               "Rank_LSTM": {"mse": 3.79e-4, "mrr": 4.17e-2, "irr": 0.68, "irr_sd": 0.60}},
    "NYSE": {"RSR_I_wiki": {"mse": 2.26e-4, "mrr": 4.58e-2, "irr": 0.79, "irr_sd": 0.34},
             "RSR_I_industry": {"mse": 2.26e-4, "mrr": 4.51e-2, "irr": 1.06, "irr_sd": 0.27},
             "Rank_LSTM": {"mse": 2.28e-4, "mrr": 3.79e-2, "irr": 0.56, "irr_sd": 0.68}},
}
CLAIM_MARGIN = 0.09   # "significant improvement (>0.09) w.r.t. IRR" (paper §5.4)


def _load(out, market, rel, graph, seeds):
    fs = [out / f"{market}_{rel}_{graph}_s{s}.npz" for s in seeds]
    if not all(f.exists() for f in fs):
        return None
    return [dict(np.load(f)) for f in fs]


def _irr(m):
    return float(m["test_top1"].sum())


def _mse(m):
    return float(m["test_se"].sum() / m["test_cnt"].sum())


def _mrr(m):
    return float(m["test_rr"].mean())


def summarise(cfg):
    out = ROOT / cfg["experiment"]["output_dir"]
    rel, rel2 = cfg["relation_primary"], cfg["relation_secondary"]
    s_d1, s_e = range(cfg["seeds_d1"]), range(cfg["seeds_edge"])
    K = cfg["placebos"]["K"]
    inf = cfg["inference"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    tests, desc, lines = [], {}, []
    for market in cfg["markets"]:
        real, none = _load(out, market, rel, "real", s_d1), _load(out, market, rel, "none", s_d1)
        if real is None or none is None:
            lines.append(f"## {market}: real/none incomplete")
            continue
        # H1/H2: D1 on IRR, paired t-test over seeds (seed s of real with seed s of L0)
        Ar = np.stack([m["test_top1"] for m in real]); Bn = np.stack([m["test_top1"] for m in none])
        diff = Ar.sum(1) - Bn.sum(1)
        p = infer.seed_t(Ar, Bn, paired=True)
        tw = inf["D1"]["secondary"]["twoway_seed_day_bootstrap"]
        tests.append({"id": f"D1_{market}", "market": market, "quantity": "D1_IRR", "estimate": float(diff.mean()),
                      "sd_seed_diff": float(diff.std(ddof=1)), "n_seeds": int(len(diff)), "p": p,
                      "secondary_twoway_p": infer.twoway_bootstrap(Ar, Bn, tw["block_length"], tw["repetitions"], rng, paired=True),
                      "descr_day_bootstrap_p_MISCALIBRATED": infer.day_bootstrap(Ar, Bn, 20, 1000, rng),
                      "descr_5seed": {"mean_diff": float(diff[:5].mean()), "p_paired_t": infer.seed_t(Ar[:5], Bn[:5], paired=True)}})
        # H3/H4: edge on IRR (placebo rank, seeds 0..seeds_edge-1)
        T_real = np.mean([_irr(m) for m in real[:len(s_e)]])
        T_plac = []
        for k in range(K):
            pk = _load(out, market, rel, f"perm{k:03d}", s_e)
            if pk is not None:
                T_plac.append(np.mean([_irr(m) for m in pk]))
        T_plac = np.array(T_plac)
        if len(T_plac) == K:
            pe = float((1 + (T_plac >= T_real).sum()) / (K + 1))
            tests.append({"id": f"edge_{market}", "market": market, "quantity": "edge_IRR",
                          "estimate": float(T_real - T_plac.mean()), "p": pe,
                          "placebos_at_least_real": int((T_plac >= T_real).sum())})
        # descriptive
        T_none_e = np.mean([_irr(m) for m in none[:len(s_e)]])
        dd = {"real": {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in
                       {"irr": [_irr(m) for m in real], "mse": [_mse(m) for m in real], "mrr": [_mrr(m) for m in real]}.items()},
              "none": {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in
                       {"irr": [_irr(m) for m in none], "mse": [_mse(m) for m in none], "mrr": [_mrr(m) for m in none]}.items()},
              "irr_seeds0_4": {"real": (float(np.mean([_irr(m) for m in real[:5]])), float(np.std([_irr(m) for m in real[:5]], ddof=1))),
                               "none": (float(np.mean([_irr(m) for m in none[:5]])), float(np.std([_irr(m) for m in none[:5]], ddof=1)))},
              "best_epochs_real": [int(m["best_epoch"]) for m in real],
              "best_epochs_none": [int(m["best_epoch"]) for m in none],
              "n_placebos": int(len(T_plac))}
        if len(T_plac):
            dd["placebo_irr"] = {"mean": float(T_plac.mean()), "sd": float(T_plac.std(ddof=1)) if len(T_plac) > 1 else None,
                                 "min": float(T_plac.min()), "max": float(T_plac.max()), "T_real": float(T_real),
                                 "T_none": float(T_none_e),
                                 "share_beating_none_by_margin": float((T_plac - T_none_e > CLAIM_MARGIN).mean())}
            for nm, fn in (("mse", _mse), ("mrr", _mrr)):
                Tr = np.mean([fn(m) for m in real[:len(s_e)]])
                Tp = np.array([np.mean([fn(m) for m in _load(out, market, rel, f"perm{k:03d}", s_e)])
                               for k in range(K) if _load(out, market, rel, f"perm{k:03d}", s_e) is not None])
                better = (Tp <= Tr) if nm == "mse" else (Tp >= Tr)
                dd[f"edge_{nm}"] = {"real": float(Tr), "placebo_mean": float(Tp.mean()),
                                    "p_rank": float((1 + better.sum()) / (len(Tp) + 1))}
        dm = np.mean([m["test_se"] / m["test_cnt"] for m in none], 0) - np.mean([m["test_se"] / m["test_cnt"] for m in real], 0)
        pm, cim = bootstrap_p_one_sided(dm, 20, 1000, rng)  # descriptive (MSE, deterministic-day view)
        dd["D1_mse_daily"] = {"estimate": float(dm.mean()), "ci90": cim, "p": pm}
        s_ind = range(cfg["seeds_industry"])
        ind = _load(out, market, rel2, "real", s_ind)
        if ind is not None:
            Ai = np.stack([m["test_top1"] for m in ind]); Bi = Bn[:len(s_ind)]
            dd["D1_industry_IRR"] = {"mean_diff": float((Ai.sum(1) - Bi.sum(1)).mean()), "p_paired_t": infer.seed_t(Ai, Bi, paired=True),
                                     "irr": (float(np.mean([_irr(m) for m in ind])), float(np.std([_irr(m) for m in ind], ddof=1)))}
        desc[market] = dd
    if tests:
        for t, a in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = a
            t["verdict"] = "supported" if a <= inf["multiplicity"]["alpha"] else "not detected"
    complete = len(tests) == 2 * len(cfg["markets"])
    lines = ["# RSR replication — results", "",
             f"Family complete: {complete} ({len(tests)} of {2 * len(cfg['markets'])} tests). Holm over the family present.", "",
             "| test | estimate (IRR) | p | p (Holm) | verdict |", "|---|---|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['id']} | {t['estimate']:+.3f} | {t['p']:.4f} | {t.get('p_holm', float('nan')):.4f} | {t.get('verdict', '')} |")
    lines += ["", "## Reproduction check (R0, descriptive) — ours (mean ± sd over seeds) vs paper", ""]
    for market, dd in desc.items():
        P = PAPER[market]
        r5, n5 = dd["irr_seeds0_4"]["real"], dd["irr_seeds0_4"]["none"]
        gap = abs(r5[0] - P["RSR_I_wiki"]["irr"]) > P["RSR_I_wiki"]["irr_sd"]
        lines += [f"**{market}** — seeds 0–4: RSR_I-Wiki IRR {r5[0]:.2f} ± {r5[1]:.2f}, no-relation {n5[0]:.2f} ± {n5[1]:.2f}"
                  f" → {'REPRODUCTION GAP (>1 paper sd)' if gap else 'within 1 paper sd'}. All seeds: "
                  f"RSR_I-Wiki IRR {dd['real']['irr'][0]:.2f} ± {dd['real']['irr'][1]:.2f} "
                  f"(paper {P['RSR_I_wiki']['irr']:.2f} ± {P['RSR_I_wiki']['irr_sd']:.2f}); "
                  f"no-relation IRR {dd['none']['irr'][0]:.2f} ± {dd['none']['irr'][1]:.2f} "
                  f"(paper Rank_LSTM {P['Rank_LSTM']['irr']:.2f} ± {P['Rank_LSTM']['irr_sd']:.2f}); "
                  f"MSE {dd['real']['mse'][0]:.3e} vs {dd['none']['mse'][0]:.3e} (paper {P['RSR_I_wiki']['mse']:.2e} vs {P['Rank_LSTM']['mse']:.2e}); "
                  f"MRR {dd['real']['mrr'][0]:.4f} vs {dd['none']['mrr'][0]:.4f}", ""]
    lines += ["## Descriptive detail", "", "```", json.dumps(desc, indent=1), "```"]
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps({"tests": tests, "descriptive": desc}, indent=1))
    print("\n".join(lines))
