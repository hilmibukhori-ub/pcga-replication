"""Analysis for Revision 1 (PREREG_REV1.md). Written before any Revision-1 model was trained.

Study S (RSR, size-stratified placebos):
  T(G) = test IRR at seed 0, validation-selected epoch (as the parent H3/H4; rsr_replica_report._irr).
  H-S_m: p = (1 + #{k: T(P_k) >= T(real)}) / (K + 1), K = 99; Holm over {NASDAQ, NYSE}.
  T(real) = parent real s0, unless the determinism re-run differs by more than 1e-9 (then the re-run).
Study K (RT-GCN U, K = 99):
  T(G) = IRR-1 at the validation-selected epoch, seed 0, all 172 test days (as the parent H-C).
  H-C99_c over perm000-perm098; Holm over {NASDAQ_U, NYSE_U}.
Completion rule: a family is tested only when all of its models exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rsr_replica_report import _irr
from .rtgcn_report import holm, metric, sel
from .rtgcn_report import load as load_main
from .rtgcn_ext_report import load as load_ext

ROOT = Path(__file__).resolve().parents[1]


def rank_test(Tr, Tp):
    Tp = np.asarray(Tp, float)
    n = int((Tp >= Tr).sum())
    return {"T_real": float(Tr), "placebo_mean": float(Tp.mean()), "placebo_sd": float(Tp.std(ddof=1)),
            "n_at_least_real": n, "rank_of_real": n + 1, "n_graphs": len(Tp) + 1,
            "p_rank": float((1 + n) / (len(Tp) + 1))}


def edge_shift80(K, alpha, reps=40000, seed=20261116):
    """Shift of the real graph (placebo SDs) giving 80% power for the one-sided rank test (as power_posthoc.py)."""
    rng = np.random.default_rng(seed)
    pl, z = rng.standard_normal((reps, K)), rng.standard_normal(reps)
    lo, hi = 0.0, 6.0
    for _ in range(30):
        mid = (lo + hi) / 2
        p = (1 + (pl >= (z + mid)[:, None]).sum(1)) / (K + 1)
        lo, hi = (mid, hi) if (p <= alpha).mean() < 0.8 else (lo, mid)
    return hi


def _npz(f):
    return dict(np.load(f)) if f.exists() else None


def study_S(cfg):
    S = cfg["study_S"]
    import yaml
    parent = yaml.safe_load((ROOT / S["parent_config"]).read_text())
    pout, out, rel, K = ROOT / parent["experiment"]["output_dir"], ROOT / S["output_dir"], S["relation"], S["K"]
    res, tests = {}, []
    for m in S["markets"]:
        par_real = _npz(pout / f"{m}_{rel}_real_s0.npz")
        rerun = _npz(out / f"{m}_{rel}_real_s0.npz")
        pl = [_npz(out / f"{m}_{rel}_sizebin{k:03d}_s0.npz") for k in range(K)]
        c = {"complete": bool(par_real is not None and rerun is not None and all(p is not None for p in pl))}
        if c["complete"]:
            diff = abs(_irr(rerun) - _irr(par_real))
            c["determinism"] = {"parent_T_real": _irr(par_real), "rerun_T_real": _irr(rerun), "abs_diff": diff,
                                "reproduced": bool(diff <= 1e-9)}
            Tr = _irr(par_real) if diff <= 1e-9 else _irr(rerun)
            c["H-S"] = rank_test(Tr, [_irr(p) for p in pl])
            par_pl = [_npz(pout / f"{m}_{rel}_perm{k:03d}_s0.npz") for k in range(parent["placebos"]["K"])]
            if all(p is not None for p in par_pl):
                c["parent_edge_same_Treal"] = rank_test(Tr, [_irr(p) for p in par_pl])
            tests.append((m, c["H-S"]["p_rank"]))
        res[m] = c
    if len(tests) == len(S["markets"]):
        for (m, _), adj in zip(tests, holm([p for _, p in tests])):
            res[m]["H-S"]["p_holm"] = adj
            res[m]["H-S"]["verdict"] = "supported" if adj <= cfg["alpha"] else "not detected"
    return res


def study_K(cfg, roots=None):
    """roots: optional {cell: output dir} override (e.g. an rsync copy of the hyperion outputs)."""
    Kc = cfg["study_K"]
    K = Kc["K"]
    res, tests = {}, []
    for cell, spec in Kc["cells"].items():
        out = Path(roots[cell]) if roots else ROOT / spec["output_dir"]
        if cell == "NASDAQ_U":
            get = lambda g: load_main("val", "paper", g, [0], out)          # noqa: E731
        else:
            get = lambda g, cell=cell: load_ext(out, cell, "val", g, [0])   # noqa: E731
        real, pl = get("real"), [get(f"perm{k:03d}") for k in range(K)]
        c = {"complete": bool(real is not None and all(p is not None for p in pl))}
        if c["complete"]:
            r = real[0]
            Tr = metric(r, sel(r, "val"))
            Tp = [metric(p[0], sel(p[0], "val")) for p in pl]
            c["H-C99"] = rank_test(Tr, Tp)
            c["H-C19_recomputed"] = rank_test(Tr, Tp[:19])
            sd = c["H-C99"]["placebo_sd"]
            c["detectable_80"] = {f"alpha_{a}": {"shift_sd": s, "shift_irr1": s * sd}
                                  for a in (0.05, 0.025) for s in [edge_shift80(K, a)]}
            if cell == "NYSE_U":
                ex = Kc["secondary"]["nyse_excl_day"]
                assert ex == 89
                c["excl_day89_descriptive"] = rank_test(metric(r, sel(r, "val"), "irr1", "pub"),
                                                        [metric(p[0], sel(p[0], "val"), "irr1", "pub") for p in pl])
            tests.append((cell, c["H-C99"]["p_rank"]))
        res[cell] = c
    if len(tests) == len(Kc["cells"]):
        for (cell, _), adj in zip(tests, holm([p for _, p in tests])):
            res[cell]["H-C99"]["p_holm"] = adj
            res[cell]["H-C99"]["verdict"] = "supported" if adj <= cfg["alpha"] else "not detected"
    return res


def summarise(cfg, roots=None, write=True):
    res = {"study_S": study_S(cfg), "study_K": study_K(cfg, roots)}
    if not write:
        return res
    (ROOT / "outputs/rev1_results.json").write_text(json.dumps(res, indent=1))
    L = ["# Revision 1 robustness — results (PREREG_REV1.md)", "",
         "Post hoc in motivation, pre-registered before any Revision-1 model was trained.", "",
         "## Study S — RSR, industry x size-bin placebos (K = 99, Holm over 2 markets)", ""]
    for m, c in res["study_S"].items():
        if not c["complete"]:
            L.append(f"- **{m}**: INCOMPLETE — no test computed (completion rule).")
            continue
        h, d = c["H-S"], c["determinism"]
        L.append(f"- **H-S {m}**: real IRR {h['T_real']:+.3f}, size-bin placebo mean {h['placebo_mean']:+.3f} "
                 f"(sd {h['placebo_sd']:.3f}); real ranks {h['rank_of_real']}/{h['n_graphs']}; "
                 f"p {h['p_rank']:.3f}, Holm {h.get('p_holm', float('nan')):.3f} → {h.get('verdict', '-')}")
        if "parent_edge_same_Treal" in c:
            pe = c["parent_edge_same_Treal"]
            L.append(f"  - parent within-industry null, same T(real): rank {pe['rank_of_real']}/{pe['n_graphs']}, p {pe['p_rank']:.3f}")
        L.append(f"  - determinism: re-run real IRR {d['rerun_T_real']:+.6f} vs parent {d['parent_T_real']:+.6f} "
                 f"(|diff| {d['abs_diff']:.2e}, {'reproduced' if d['reproduced'] else 'NOT reproduced → re-run used; record R1 deviation'})")
    L += ["", "## Study K — RT-GCN(U), K = 99 (Holm over 2 cells)", ""]
    for cell, c in res["study_K"].items():
        if not c["complete"]:
            L.append(f"- **{cell}**: INCOMPLETE — no test computed (completion rule).")
            continue
        h, o = c["H-C99"], c["H-C19_recomputed"]
        L.append(f"- **H-C99 {cell}**: real IRR-1 {h['T_real']:+.3f}, placebo mean {h['placebo_mean']:+.3f} "
                 f"(sd {h['placebo_sd']:.3f}); rank {h['rank_of_real']}/{h['n_graphs']} (K=19: {o['rank_of_real']}/{o['n_graphs']}); "
                 f"p {h['p_rank']:.3f}, Holm {h.get('p_holm', float('nan')):.3f} → {h.get('verdict', '-')}")
        dd = c["detectable_80"]
        L.append(f"  - 80% power needs a shift of {dd['alpha_0.05']['shift_sd']:.2f} placebo sd = {dd['alpha_0.05']['shift_irr1']:.2f} IRR-1 "
                 f"(α 0.05); {dd['alpha_0.025']['shift_sd']:.2f} sd = {dd['alpha_0.025']['shift_irr1']:.2f} (Holm first step)")
        if "excl_day89_descriptive" in c:
            e = c["excl_day89_descriptive"]
            L.append(f"  - descriptive, day 89 excluded: rank {e['rank_of_real']}/{e['n_graphs']}, p {e['p_rank']:.3f}")
    (ROOT / "RESULTS_REV1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return res
