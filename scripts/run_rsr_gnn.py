"""RSR GCN/MLP robustness (exploratory). Usage:
  python scripts/run_rsr_gnn.py --mode confirm --market NYSE --seed 0      # train (resumable)
  python scripts/run_rsr_gnn.py --summarise                               # report
  python scripts/run_rsr_gnn.py --mode dev --market NYSE --seed 0 --smoke  # quick dev check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rsr_gnn as g  # noqa: E402
from dapilot.power_v2 import rank_test_p  # noqa: E402
from dapilot.rsr_pipeline import bootstrap_p_one_sided  # noqa: E402


def load(out, market, seeds):
    """{graph: {loss: (S, T) array}} for graphs present for all seeds."""
    res = {}
    for f in sorted(out.glob(f"{market}_s*_*.npz")):
        s, name = f.stem.split("_", 2)[1:]
        res.setdefault(name, {})[int(s[1:])] = dict(np.load(f))
    return {k: v for k, v in res.items() if set(v) >= set(range(seeds))}


def summarise(gc, out):
    lines = ["# RSR GCN/MLP robustness (EXPLORATORY, descriptive only)", "",
             "PREREG_RSR.md §5/§9.4; spec in configs/rsr_gnn.yaml; DEVIATIONS G1. Not confirmatory. "
             "Positive = graph model has lower loss. Losses averaged over seeds before differencing. "
             f"Edge rank p uses K = {gc['k_l3']} (min p = {1/(gc['k_l3']+1):.2f}); intervals: circular block bootstrap L=20, 90%.", ""]
    rng = np.random.default_rng(0)
    js = {}
    for market in g.MARKETS:
        R = load(out, market, gc["seeds"])
        if "L0_none" not in R or "L4_real" not in R:
            lines.append(f"## {market}: incomplete ({len(R)} graphs)"); continue
        mean = {k: {l: np.mean([v[s][l] for s in range(gc["seeds"])], 0) for l in ("mse", "mse_raw", "mae", "ic")}
                for k, v in R.items()}
        lev = lambda p: [k for k in mean if k.startswith(p)]  # noqa: E731
        L3 = lev("L3_industry")
        lvl = lambda p, l="mse": np.mean([mean[k][l] for k in lev(p)], 0)  # noqa: E731
        lines += [f"## {market}", "", f"Graphs complete: {len(R)} (L3: {len(L3)}, L1: {len(lev('L1_er'))}, L2: {len(lev('L2_degree'))}); "
                  f"test {R['L0_none'][0]['test_first']} → {R['L0_none'][0]['test_last']}", ""]
        d1 = mean["L0_none"]["mse"] - mean["L4_real"]["mse"]
        p1, ci1 = bootstrap_p_one_sided(d1, 20, 1000, rng)
        rows = [("D1 (MLP − GCN-Wiki)", d1, p1, ci1, "bootstrap")]
        if L3:
            l3each = np.array([mean[k]["mse"].mean() for k in L3])
            ed = lvl("L3_industry") - mean["L4_real"]["mse"]
            pr = float(rank_test_p(np.array([mean["L4_real"]["mse"].mean()]), l3each[:, None])[0])
            _, cie = bootstrap_p_one_sided(ed, 20, 1000, rng)
            rows.append(("edge (L3 mean − Wiki)", ed, pr, cie, f"rank, {int(round(pr*(len(L3)+1)))-1}/{len(L3)} placebos better"))
        base = mean["L4_real"]["mse"].mean()
        lines += ["| quantity | estimate | rel. to L4 MSE | 90% interval | p | test |", "|---|---|---|---|---|---|"]
        for nm, d, p, ci, how in rows:
            lines.append(f"| {nm} | {d.mean():+.2e} | {d.mean()/base*100:+.4f}% | [{ci[0]:+.2e}, {ci[1]:+.2e}] | {p:.3f} | {how} |")
        if lev("L1_er") and lev("L2_degree") and L3:
            st = {"density": mean["L0_none"]["mse"] - lvl("L1_er"), "degree": lvl("L1_er") - lvl("L2_degree"),
                  "industry": lvl("L2_degree") - lvl("L3_industry"), "edge": lvl("L3_industry") - mean["L4_real"]["mse"]}
            lines += ["", "Ladder (point estimates): " + ", ".join(f"{k} {v.mean():+.2e}" for k, v in st.items())]
        lines += ["", "Per seed (D1 / edge, MSE): " + "; ".join(
            f"s{s}: {R['L0_none'][s]['mse'].mean()-R['L4_real'][s]['mse'].mean():+.2e} / "
            + (f"{np.mean([R[k][s]['mse'].mean() for k in L3])-R['L4_real'][s]['mse'].mean():+.2e}" if L3 else "n/a")
            for s in range(gc["seeds"]))]
        lines += ["", "IC (seed-avg prediction losses, mean daily Pearson): MLP "
                  f"{np.nanmean(mean['L0_none']['ic']):.4f}, GCN-Wiki {np.nanmean(mean['L4_real']['ic']):.4f}"
                  + (f", GCN-L3 mean {np.nanmean(lvl('L3_industry','ic')):.4f}" if L3 else ""), ""]
        sec = {l: (mean["L0_none"][l] - mean["L4_real"][l]).mean() for l in ("mse_raw", "mae")}
        sece = {l: (lvl("L3_industry", l) - mean["L4_real"][l]).mean() for l in ("mse_raw", "mae")} if L3 else {}
        lines += [f"Secondary losses D1: mse_raw {sec['mse_raw']:+.2e}, mae {sec['mae']:+.2e}"
                  + (f"; edge: mse_raw {sece['mse_raw']:+.2e}, mae {sece['mae']:+.2e}" if L3 else ""), ""]
        info = {k: {s: {"wd": float(v[s]["wd"]), "epoch": int(v[s]["best_epoch"])} for s in v} for k, v in R.items()}
        eps = [i["epoch"] for v in info.values() for i in v.values()]
        lines += [f"Best epoch: median {np.median(eps):.0f}, min {min(eps)}, max {max(eps)} (max_epochs {gc['experts']['max_epochs']})", ""]
        js[market] = {"rows": [(r[0], float(r[1].mean()), r[2], r[3]) for r in rows], "info": info}
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(js, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/rsr_gnn.yaml"))
    ap.add_argument("--mode", choices=["dev", "confirm"])
    ap.add_argument("--market")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--only", nargs="*", help="graph-name prefixes to train (for splitting work)")
    ap.add_argument("--smoke", action="store_true", help="dev: 3 epochs, K_L3=1, separate output dir")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    gc = yaml.safe_load(Path(a.config).read_text())
    cfg = yaml.safe_load((ROOT / gc["prereg_config"]).read_text())
    out = ROOT / gc["output_dir"]
    torch.set_num_threads(1)
    if a.summarise:
        summarise(gc, out)
    else:
        ecfg, k = dict(gc["experts"]), gc["k_l3"]
        if a.mode == "dev":
            out = ROOT / "outputs/rsr_gnn_dev"
        if a.smoke:
            assert a.mode == "dev"
            ecfg["max_epochs"], k = 3, 1
        g.run(cfg, ecfg, a.market, a.mode, a.seed, k, out, log=lambda s: print(s, flush=True), only=a.only)
