"""EXPLORATORY, post hoc (DEVIATIONS RX-S1): extension H-A/H-B/H-C recomputed with test day 89 excluded.
Written after the confirmatory read; the confirmatory verdicts (all 172 days) are unchanged by this file."""
import sys, json, pathlib, numpy as np, yaml
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dapilot.rtgcn_report import sel, metric, paired
from dapilot.rtgcn_ext_report import load
cfg = yaml.safe_load(open(ROOT / "configs/rtgcn_ext.yaml")); out = ROOT / cfg["output_dir"]
res = {}
for cell in cfg["cells"]:
    S, K = range(cfg["seeds"][cell]), cfg["placebos"][cell]
    real, none = load(out, cell, "val", "real", S), load(out, cell, "val", "none", S)
    ha = paired([metric(r, sel(r, "test"), "irr1", "pub") for r in real], [metric(r, sel(r, "val"), "irr1", "pub") for r in real])
    hb = paired([metric(r, sel(r, "val"), "irr1", "pub") for r in real], [metric(n, sel(n, "val"), "irr1", "pub") for n in none])
    Tr = metric(real[0], sel(real[0], "val"), "irr1", "pub")
    Tp = np.array([metric(p[0], sel(p[0], "val"), "irr1", "pub") for p in [load(out, cell, "val", f"perm{k:03d}", [0]) for k in range(K)]])
    hc = {"T_real": Tr, "placebo_mean": float(Tp.mean()), "n_at_least_real": int((Tp >= Tr).sum()), "p_rank": float((1 + (Tp >= Tr).sum()) / (K + 1))}
    sd = float(np.std([metric(r, sel(r, "val"), "irr1", "pub") for r in real], ddof=1))
    res[cell] = {"H-A_excl89": ha, "H-B_excl89": hb, "H-C_excl89": hc, "between_seed_sd_excl89": sd}
    print(f"{cell}: H-A {ha['estimate']:+.3f} [{ha['ci95'][0]:+.3f},{ha['ci95'][1]:+.3f}] p {ha['p']:.3f} | "
          f"H-B {hb['estimate']:+.3f} [{hb['ci95'][0]:+.3f},{hb['ci95'][1]:+.3f}] p {hb['p']:.3f} ({hb['n_pos']}/{hb['n']}) | "
          f"H-C {hc['n_at_least_real']}/{K} >= real, p {hc['p_rank']:.2f} | sd {sd:.2f}")
json.dump(res, open(out / "sensitivity_excl_day89.json", "w"), indent=1)
