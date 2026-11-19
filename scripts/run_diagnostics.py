"""Run post-hoc hard/soft and structural-information diagnostics on a completed pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.diagnostics import run_diagnostics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = run_diagnostics(args.run_dir)
    for name, summary in result["summaries"].items():
        print(f"{name}: {summary['mean']:+.6f} "
              f"[{summary['t_ci_low']:+.6f}, {summary['t_ci_high']:+.6f}], "
              f"p={summary['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
