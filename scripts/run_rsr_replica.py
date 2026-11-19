"""Re-test of the RSR (Feng et al. 2019) relation claim with retrained placebos.
  python scripts/run_rsr_replica.py --mode dev --market NASDAQ --graph real --seeds 0      # dev sanity run
  python scripts/run_rsr_replica.py --mode confirm --market NYSE --graph perm --k 0:25 --seeds 0:3
  python scripts/run_rsr_replica.py --summarise
Every model is one npz; existing files are skipped (resumable). Confirm mode refuses to run unless
the hashes in PREREG_REPLICATION_FREEZE.md match."""
import argparse, hashlib, re, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rsr_replica as rr  # noqa: E402

FREEZE = ROOT / "PREREG_REPLICATION_FREEZE.md"


def verify_freeze():
    if not FREEZE.exists():
        sys.exit("confirm mode: PREREG_REPLICATION_FREEZE.md missing (prereg not frozen)")
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", FREEZE.read_text())
    if not rows:
        sys.exit("no hashes in freeze file")
    for p, h in rows:
        got = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        if got != h:
            sys.exit(f"FREEZE MISMATCH {p}: {got} != {h}")


def rng_range(s):
    a, b = (int(x) for x in s.split(":")) if ":" in s else (int(s), int(s) + 1)
    return range(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "confirm"])
    ap.add_argument("--market")
    ap.add_argument("--relation", default=None)
    ap.add_argument("--graph", choices=["real", "none", "perm"])
    ap.add_argument("--k", default="0", help="placebo index range a:b")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=None, help="dev only")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/rsr_replica.yaml").read_text())
    if a.summarise:
        from dapilot.rsr_replica_report import summarise
        return summarise(cfg)
    if a.mode == "confirm":
        verify_freeze()
        assert a.epochs is None, "epochs fixed in confirm mode"
    out = ROOT / cfg["experiment"]["output_dir" if a.mode == "confirm" else "dev_output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    mc = cfg["markets"][a.market]
    rel_name = a.relation or cfg["relation_primary"]
    mi = list(cfg["markets"]).index(a.market)
    data = rr.Data(a.market, mc["emb"], mc["seq"], a.mode, cfg["training"]["steps"])
    rel = rr.load_relation(a.market, rel_name)
    R, N = rel.shape[2], rel.shape[0]
    E_real = rr.to_edges(rel)
    del rel
    labels = rr.industry_labels(a.market)
    jobs = []
    if a.graph == "real":
        jobs = [("real", E_real)]
    elif a.graph == "none":
        jobs = [("none", rr.self_only_edges(N, R))]
    else:
        for k in rng_range(a.k):
            pi = rr.within_label_perm(labels, np.random.default_rng([cfg["experiment"]["seed"], mi, 7, k]))
            jobs.append((f"perm{k:03d}", rr.permute_edges(E_real, pi)))
    for name, E in jobs:
        for s in rng_range(a.seeds):
            f = out / f"{a.market}_{rel_name}_{name}_s{s}.npz"
            if f.exists():
                continue
            t = time.time()
            seed = cfg["experiment"]["seed"] + 1000 * mi + s   # common random numbers across graphs
            res = rr.run_model(data, E, R, mc["unit"], mc["alpha"], seed,
                               epochs=a.epochs or cfg["training"]["epochs"], lr=cfg["training"]["lr"],
                               eval_test=True)  # dev "test" = the original validation year (no 2017 rows loaded)
            arrs = {f"{part}_{k}": v for part in ("val", "test") for k, v in res[part].items()}
            np.savez(f, **arrs, best_epoch=res["best_epoch"], val_loss=res["val_loss"], seed=seed)
            sv, st = rr.summary(res["val"]), rr.summary(res["test"])
            print(f"{f.stem}: ep*={res['best_epoch']} {time.time()-t:.0f}s val irr={sv['irr_sum']:+.3f} "
                  f"mse={sv['mse']:.3e} | test irr={st['irr_sum']:+.3f} mrr={st['mrrt']:.4f} mse={st['mse']:.3e}",
                  flush=True)


if __name__ == "__main__":
    main()
