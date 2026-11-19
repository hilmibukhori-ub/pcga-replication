"""P1 power on the real RSR Wiki topology.

    python scripts/run_power_wiki.py --market NASDAQ --reps 0:25
    python scripts/run_power_wiki.py --summarise
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot.power_wiki import evaluate, run_cell, summarise  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/power_wiki_p1.yaml"))
    ap.add_argument("--market")
    ap.add_argument("--reps")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = ROOT / cfg["experiment"]["output_dir"]
    if a.summarise:
        summarise(evaluate(cfg, out), cfg, out)
        print((out / "REPORT.md").read_text())
    else:
        lo, hi = (int(v) for v in a.reps.split(":")) if a.reps else (0, cfg["experiment"]["reps"][a.market])
        run_cell(cfg, a.market, (lo, hi), out, log=lambda s: print(s, flush=True))
