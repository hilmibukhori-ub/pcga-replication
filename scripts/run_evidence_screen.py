"""Run Stage 9 oracle-supervised evidence-family screening."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.discovery import run_oracle_screen  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = run_oracle_screen(args.source_run, args.output_dir)
    print("passed families:", result["passed_families"])
    for row in result["summary"]:
        if row["subset"] == "all" and row["metric"] == "risk_diff":
            print(f"{row['family']}: {row['mean']:+.6f} "
                  f"[{row['t_ci_low']:+.6f}, {row['t_ci_high']:+.6f}], "
                  f"p_holm={row['p_holm']:.4f}, {row['screening']}")


if __name__ == "__main__":
    main()
