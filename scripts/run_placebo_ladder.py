"""Run the placebo-ladder synthetic screen (ridge experts, ~6 min CPU)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.ladder import load_config, run  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs/placebo_ladder.yaml")
    out = ROOT / cfg["experiment"]["output_dir"]
    run(cfg, out)
    print((out / "REPORT.md").read_text())
