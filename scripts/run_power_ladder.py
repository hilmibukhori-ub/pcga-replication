"""Power analysis for the placebo ladder at RSR / CRSP panel sizes.

usage: python scripts/run_power_ladder.py [config] [--cells a,b] [--reps N] [--out DIR]
       python scripts/run_power_ladder.py --merge   # combine <output_dir>/cells/* into one report
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from dapilot.power import load_config, run, summarise  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(ROOT / "configs/power_ladder.yaml"))
    ap.add_argument("--cells")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--out")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out) if a.out else ROOT / cfg["experiment"]["output_dir"]
    if a.merge:
        dirs = sorted((out / "cells").iterdir())
        df = pd.concat([pd.read_csv(d / "replicates.csv") for d in dirs])
        dg = pd.concat([pd.read_csv(d / "placebo_diagnostics.csv") for d in dirs])
        df.to_csv(out / "replicates.csv", index=False)
        dg.to_csv(out / "placebo_diagnostics.csv", index=False)
        summarise(df, dg, cfg, out)
        print((out / "REPORT.md").read_text())
        sys.exit()
    run(cfg, out, cells=a.cells.split(",") if a.cells else None, reps=a.reps, log=lambda s: print(s, flush=True))
    print((out / "REPORT.md").read_text())
