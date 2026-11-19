"""RSR ladder (PREREG_RSR.md).

    python scripts/run_rsr.py --mode dev [--k-l3 5]     # smoke test: train 2013-14, val 2015, pseudo-test 2016
    python scripts/run_rsr.py --mode confirm --read-test-once
                                                        # the one confirmatory run on 2017 (freeze-checked)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rsr_pipeline as rp  # noqa: E402


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "confirm"], required=True)
    ap.add_argument("--k-l3", type=int, help="dev only: fewer L3 placebos for a quick smoke test")
    ap.add_argument("--read-test-once", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/rsr_prereg.yaml").read_text())
    if a.mode == "confirm":
        if not a.read_test_once or a.k_l3:
            sys.exit("confirm mode needs --read-test-once and the frozen K_L3")
        out = ROOT / cfg["experiment"]["output_dir"]
        if (out / "results.json").exists():
            sys.exit(f"{out}/results.json exists: the test period has already been read once")
    else:
        out = ROOT / "outputs/rsr_dev"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    res = {m: rp.run_market(cfg, m, a.mode, k_l3=a.k_l3) for m in cfg["data"]["markets"]}
    conf = rp.confirmatory(cfg, res, np.random.default_rng([cfg["experiment"]["seed"], 7]))
    summary = {"mode": a.mode, "started": started, "k_l3": a.k_l3 or cfg["graphs"]["placebos"]["L3_industry"],
               "confirmatory": conf,
               "markets": {m: {k: (v if not isinstance(v, dict) else
                                   {q: (float(np.mean(x)) if isinstance(x, np.ndarray) and q != "L3_each_mean" else x)
                                    for q, x in v.items()})
                               for k, v in r.items()} for m, r in res.items()}}
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=_js))
    np.savez_compressed(out / "daily.npz", **{f"{m}_{loss}_{q}": res[m][loss][q] for m in res
                                              for loss in ("mse", "mse_raw", "mae")
                                              for q in ("D1", "density", "degree", "industry", "edge", "L3_each_mean")})
    print(f"mode={a.mode} k_l3={summary['k_l3']}")
    for r in conf:
        print(f"{r['id']} {r['market']:6s} {r['quantity']:4s} est={r['estimate']:+.3e} ci90=[{r['ci90'][0]:+.2e},{r['ci90'][1]:+.2e}] "
              f"p={r['p']:.3f} p_holm={r['p_holm']:.3f} {r['verdict']}")
    for m, r in summary["markets"].items():
        print(m, "test", r["test_dates"], r["test_days"], "days; alpha", r["alpha"], "L3 overlap", round(r["l3_overlap"], 2), "IC", r["ic"])
