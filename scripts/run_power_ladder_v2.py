"""Power analysis v2 (inference validity).

usage: python scripts/run_power_ladder_v2.py --cell rsr_nasdaq --reps 0:40   # one process
       python scripts/run_power_ladder_v2.py --summarise                     # evaluate + report
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from dapilot.power_v2 import evaluate, run_cell, summarise  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/power_ladder_v2.yaml"))
    ap.add_argument("--cell")
    ap.add_argument("--reps")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = ROOT / cfg["experiment"]["output_dir"]
    if a.summarise:
        summarise(evaluate(cfg, out), cfg, out)
        print((out / "REPORT.md").read_text())
    else:
        lo, hi = (int(v) for v in a.reps.split(":")) if a.reps else (0, cfg["experiment"]["reps"][a.cell])
        run_cell(cfg, a.cell, (lo, hi), out, log=lambda s: print(s, flush=True))
