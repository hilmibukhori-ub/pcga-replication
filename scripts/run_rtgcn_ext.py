"""RT-GCN extension runner (PREREG_RTGCN_EXT.md). One npz per cell/arm/graph/seed; resumable.
  python scripts/run_rtgcn_ext.py --market NYSE --variant U --mode val --graph real --seeds 0
  python scripts/run_rtgcn_ext.py --market NASDAQ --variant T --mode val --graph perm --k 3 --seeds 0
  dev rehearsal: --mode dev|devpub (never loads the released test arrays) [--epochs N] [--out DIR]
Confirm modes (val, pub) refuse to run unless PREREG_RTGCN_EXT_FREEZE.md hashes match."""
import argparse, hashlib, re, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rtgcn_replica as rt  # noqa: E402
from dapilot import rtgcn_ext as ext     # noqa: E402

FREEZE = ROOT / "PREREG_RTGCN_EXT_FREEZE.md"
CFG = yaml.safe_load((ROOT / "configs/rtgcn_ext.yaml").read_text())


def verify_freeze():
    if not FREEZE.exists():
        sys.exit("confirm mode: PREREG_RTGCN_EXT_FREEZE.md missing (not frozen)")
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", FREEZE.read_text())
    if not rows:
        sys.exit("no hashes in freeze file")
    for p, h in rows:
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            sys.exit(f"FREEZE MISMATCH {p}")


def rr(s):
    a, b = (int(x) for x in s.split(":")) if ":" in s else (int(s), int(s) + 1)
    return range(a, b)


def out_name(market, variant, mode, graph, s):
    return f"{market}_{variant}_{mode}_paper_{graph}_s{s}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["NASDAQ", "NYSE"], required=True)
    ap.add_argument("--variant", choices=["U", "T"], required=True)
    ap.add_argument("--mode", choices=["dev", "devpub", "val", "pub"], required=True)
    ap.add_argument("--graph", choices=["real", "none", "ind", "perm"], required=True)
    ap.add_argument("--k", default="0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--out", default=None, help="dev only: output dir")
    ap.add_argument("--threads", type=int, default=None, help="dev only; confirm modes use the configured count")
    ap.add_argument("--log-epochs", action="store_true")
    a = ap.parse_args()
    cell = f"{a.market}_{a.variant}"
    assert cell in CFG["cells"], f"cell {cell} not in the pre-registered extension"
    confirm = a.mode in ("val", "pub")
    if confirm:
        verify_freeze()
        assert a.epochs == CFG["epochs"] and a.out is None, "epochs/out fixed in confirm modes"
    out = ROOT / (a.out or (CFG["output_dir"] if confirm else "outputs/rtgcn_ext_dev"))
    out.mkdir(parents=True, exist_ok=True)
    if confirm:
        assert a.threads is None, "threads fixed per cell in confirm modes"
    torch.set_num_threads(a.threads or CFG["threads"][cell])
    c = CFG["config"]
    bs = CFG["batch_size"][a.market]
    data = rt.load(a.market)
    Xtr, Ptr, Ytr = data["train"]
    tr_i, va_i, te = rt.splits(a.mode)
    train, val = (Xtr, Ptr, Ytr, tr_i), (Xtr, Ptr, Ytr, va_i)
    test = (*data["test"], np.arange(len(data["test"][2]))) if confirm else (Xtr, Ptr, Ytr, te)
    names = [(f"perm{k:03d}", k) for k in rr(a.k)] if a.graph == "perm" else [(a.graph, None)]
    for name, k in names:
        todo = [s for s in rr(a.seeds) if not (out / out_name(a.market, a.variant, a.mode, name, s)).exists()]
        if not todo:
            continue
        edge, rel = ext.graph_for(a.market, a.graph, k, CFG["seed0"])
        for s in todo:
            f = out / out_name(a.market, a.variant, a.mode, name, s)
            t = time.time()
            ck = out / (f.stem + ".ckpt.pt")
            try:
                res = ext.run_ext(edge, rel, a.variant, CFG["seed0"] + s, train, val, test, epochs=a.epochs,
                                  lr=c["lr"], wd=c["wd"], bs=bs,
                                  log=(lambda x: print(x, flush=True)) if a.log_epochs else None, ckpt=ck)
                tmp = f.with_suffix(".tmp.npz")
                np.savez(tmp, **res, seed=CFG["seed0"] + s)
                tmp.rename(f)
                ck.unlink(missing_ok=True)
            except Exception:
                print(f"ERROR in {f.stem}:\n{traceback.format_exc()}", flush=True)
                continue
            # deliberately no metric in the log line (results are read only through the report)
            print(f"{f.stem}: done in {time.time()-t:.0f}s ({len(res['test_loss'])} epochs)", flush=True)


if __name__ == "__main__":
    main()
