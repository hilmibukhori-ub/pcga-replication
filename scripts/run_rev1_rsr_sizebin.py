"""Revision 1, study S runner (PREREG_REV1.md): RSR Wiki models on industry x size-bin placebos, seed 0.

  python scripts/run_rev1_rsr_sizebin.py --mode confirm --market NASDAQ --graph sizebin --k 0:17
  python scripts/run_rev1_rsr_sizebin.py --mode confirm --market NYSE --graph real          # determinism check
  python scripts/run_rev1_rsr_sizebin.py --mode dev --market NASDAQ --graph sizebin --k 0:2  # no 2017 rows

Training is the frozen RSR_I port (dapilot/rsr_replica.run_model) with the parent config and seed formula, so
the size-bin placebos share common random numbers with the parent real graph at seed 0. Confirm mode refuses to
run unless PREREG_REV1_FREEZE.md hashes match. Resumable (skips existing npz)."""
import argparse, hashlib, re, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rsr_replica as rr          # noqa: E402
from dapilot import rev1_sizebin as sb         # noqa: E402

FREEZE = ROOT / "PREREG_REV1_FREEZE.md"


def verify_freeze():
    if not FREEZE.exists():
        sys.exit("confirm mode: PREREG_REV1_FREEZE.md missing (not frozen)")
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", FREEZE.read_text())
    if not rows:
        sys.exit("no hashes in freeze file")
    for p, h in rows:
        if not (ROOT / p).exists():
            continue                                  # hyperion-only files are verified there
        got = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        if got != h:
            sys.exit(f"FREEZE MISMATCH {p}: {got} != {h}")


def rng_range(s):
    a, b = (int(x) for x in s.split(":")) if ":" in s else (int(s), int(s) + 1)
    return range(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "confirm"], required=True)
    ap.add_argument("--market", choices=["NASDAQ", "NYSE"], required=True)
    ap.add_argument("--graph", choices=["real", "sizebin"], required=True)
    ap.add_argument("--k", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="dev only")
    a = ap.parse_args()
    rev = yaml.safe_load((ROOT / "configs/rev1.yaml").read_text())["study_S"]
    cfg = yaml.safe_load((ROOT / rev["parent_config"]).read_text())
    if a.mode == "confirm":
        verify_freeze()
        assert a.epochs is None, "epochs fixed in confirm mode"
        out = ROOT / rev["output_dir"]
    else:
        out = ROOT / (rev["output_dir"].replace("_v1", "_dev"))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    mc = cfg["markets"][a.market]
    mi = list(cfg["markets"]).index(a.market)
    seed0 = cfg["experiment"]["seed"]
    rel_name = rev["relation"]
    data = rr.Data(a.market, mc["emb"], mc["seq"], a.mode, cfg["training"]["steps"])
    rel = rr.load_relation(a.market, rel_name)
    R = rel.shape[2]
    E_real = rr.to_edges(rel)
    del rel
    if a.graph == "real":
        jobs = [("real", E_real)]
    else:
        lab_sb = sb.sizebin_labels(rr.industry_labels(a.market), sb.size_measure(a.market, rev["size"]["min_days"]))
        jobs = [(f"sizebin{k:03d}", rr.permute_edges(E_real, sb.sizebin_pi(lab_sb, seed0, mi, k))) for k in rng_range(a.k)]
    s = rev["seed"]
    for name, E in jobs:
        f = out / f"{a.market}_{rel_name}_{name}_s{s}.npz"
        if f.exists():
            continue
        t = time.time()
        seed = seed0 + 1000 * mi + s                     # identical to the parent runner (common random numbers)
        res = rr.run_model(data, E, R, mc["unit"], mc["alpha"], seed,
                           epochs=a.epochs or cfg["training"]["epochs"], lr=cfg["training"]["lr"], eval_test=True)
        arrs = {f"{part}_{k}": v for part in ("val", "test") for k, v in res[part].items()}
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, **arrs, best_epoch=res["best_epoch"], val_loss=res["val_loss"], seed=seed)
        tmp.rename(f)
        sv, st = rr.summary(res["val"]), rr.summary(res["test"])
        print(f"{f.stem}: ep*={res['best_epoch']} {time.time()-t:.0f}s val irr={sv['irr_sum']:+.3f} "
              f"| test irr={st['irr_sum']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
