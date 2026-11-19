"""Run the post-hoc conditional graph-benefit predictability-ceiling analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.ceiling import run_ceiling_analysis  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = run_ceiling_analysis(args.run_dir)
    d = result["decision_summary"]
    for key in ("empirical_information_gap", "noisy_label_estimation_gap",
                "current_estimator_gap", "hard_to_soft_gap", "soft_oracle_advantage"):
        s = d[key]
        print(f"{key}: {s['mean']:+.6f} [{s['t_ci_low']:+.6f}, {s['t_ci_high']:+.6f}]")


if __name__ == "__main__":
    main()
