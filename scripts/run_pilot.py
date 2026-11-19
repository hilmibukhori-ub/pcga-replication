"""Run the synthetic pilot: all seeds in parallel, then aggregate.

    python scripts/run_pilot.py --config configs/pilot.yaml
    python scripts/run_pilot.py --config configs/pilot.yaml --seeds 0 1 --run-name smoke
    python scripts/run_pilot.py --config configs/pilot.yaml --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.config import config_hash, load_config, save_config  # noqa: E402


def _worker(args):
    cfg, seed, run_dir = args
    from dapilot.pipeline import run_seed
    t0 = time.time()
    run_seed(cfg, seed, run_dir)
    return seed, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/pilot.yaml"))
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--run-name")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.seeds:
        cfg["experiment"]["seeds"] = a.seeds
    h = config_hash(cfg)
    name = a.run_name or f"{cfg['experiment']['name']}_{h}"
    run_dir = ROOT / cfg["experiment"]["output_dir"] / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not a.aggregate_only:
        import numpy, sklearn, torch, scipy, pandas  # noqa: E401
        save_config(cfg, run_dir / "config.yaml")
        manifest = {
            "config_hash": h, "seeds": cfg["experiment"]["seeds"], "started": time.ctime(),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": numpy.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__,
        }
        n_jobs = a.n_jobs or cfg["experiment"]["n_jobs"]
        jobs = [(cfg, s, str(run_dir)) for s in cfg["experiment"]["seeds"]]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for seed, dt in ex.map(_worker, jobs):
                print(f"seed {seed} finished in {dt:.0f}s", flush=True)
        manifest["finished"] = time.ctime()
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)

    from dapilot.report import aggregate
    summary = aggregate(run_dir, cfg)
    print(f"\nResults in {run_dir}")
    for hrow in summary["hypotheses"]:
        print(f"  {hrow['id']}: {hrow['verdict']:<22} mean={hrow['mean']:+.4f}  "
              f"p_holm={hrow['p_holm']:.4f}  {hrow['hypothesis']}")


if __name__ == "__main__":
    main()
