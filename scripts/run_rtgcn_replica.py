"""RT-GCN re-test runner (one npz per arm/config/graph/seed; resumable). See PREREG_RTGCN.md.
  python scripts/run_rtgcn_replica.py --mode val --cfg paper --graph real --seeds 0:20
  python scripts/run_rtgcn_replica.py --mode pub --cfg paper --graph real --seeds 0:10
  python scripts/run_rtgcn_replica.py --mode val --cfg paper --graph perm --k 0:19 --seeds 0
  dev rehearsal: --mode dev|devpub (never loads the released test arrays) [--epochs N for smoke tests]
Confirm modes (val, pub) refuse to run unless PREREG_RTGCN_FREEZE.md hashes match."""
import argparse, hashlib, re, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rtgcn_replica as rt  # noqa: E402

FREEZE = ROOT / "PREREG_RTGCN_FREEZE.md"
CFG = yaml.safe_load((ROOT / "configs/rtgcn_replica.yaml").read_text())


def verify_freeze():
    if not FREEZE.exists():
        sys.exit("confirm mode: PREREG_RTGCN_FREEZE.md missing (not frozen)")
    rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", FREEZE.read_text())
    if not rows:
        sys.exit("no hashes in freeze file")
    for p, h in rows:
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            sys.exit(f"FREEZE MISMATCH {p}")


def rr(s):
    a, b = (int(x) for x in s.split(":")) if ":" in s else (int(s), int(s) + 1)
    return range(a, b)


def out_name(mode, cfg, graph, s, market=CFG["market"]):
    return f"{market}_{mode}_{cfg}_{graph}_s{s}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "devpub", "val", "pub"], required=True)
    ap.add_argument("--cfg", choices=list(CFG["configs"]), required=True)
    ap.add_argument("--graph", choices=["real", "none", "ind", "perm"], required=True)
    ap.add_argument("--k", default="0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--out", default=None, help="dev only: output dir")
    ap.add_argument("--log-epochs", action="store_true")
    a = ap.parse_args()
    confirm = a.mode in ("val", "pub")
    if confirm:
        verify_freeze()
        assert a.epochs == CFG["epochs"] and a.out is None, "epochs/out fixed in confirm modes"
    out = ROOT / (a.out or ("outputs/rtgcn_v1" if confirm else "outputs/rtgcn_dev"))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    m, c = CFG["market"], CFG["configs"][a.cfg]
    n_ch = rt.N_REL_CHANNELS[m]
    data = rt.load(m)
    Xtr, Ptr, Ytr = data["train"]
    tr_i, va_i, te = rt.splits(a.mode)
    train, val = (Xtr, Ptr, Ytr, tr_i), (Xtr, Ptr, Ytr, va_i)
    test = (*data["test"], np.arange(len(data["test"][2]))) if confirm else (Xtr, Ptr, Ytr, te)
    E = rt.real_edge(m)
    jobs = {"real": lambda: [("real", E)], "none": lambda: [("none", np.eye(E.shape[0]))],
            "ind": lambda: [("ind", rt.real_edge(m, channels=range(40, 136)))],
            "perm": lambda: [(f"perm{k:03d}", rt.permute_edge(E, rt.within_label_perm(
                rt.industry_labels(m), np.random.default_rng([CFG["seed0"], 7, k])))) for k in rr(a.k)]}[a.graph]()
    for name, edge in jobs:
        for s in rr(a.seeds):
            f = out / out_name(a.mode, a.cfg, name, s)
            if f.exists():
                continue
            t = time.time()
            ck = out / (f.stem + ".ckpt.pt")                # per-epoch checkpoint (resumed if present)
            try:
                res = rt.run(edge, n_ch, CFG["seed0"] + s, train, val, test, epochs=a.epochs, lr=c["lr"], wd=c["wd"],
                             bs=CFG["batch_size"], log=(lambda x: print(x, flush=True)) if a.log_epochs else None,
                             ckpt=ck)
                tmp = f.with_suffix(".tmp.npz")
                np.savez(tmp, **res, seed=CFG["seed0"] + s)
                tmp.rename(f)                               # atomic: a killed run never leaves a partial file
                ck.unlink(missing_ok=True)
            except Exception:                               # log and continue with the next model
                print(f"ERROR in {f.stem}:\n{traceback.format_exc()}", flush=True)
                continue
            tl = res["test_loss"]; e_t = int(np.argmin(tl))
            msg = f"{f.stem}: {time.time()-t:.0f}s | test-loss-selected ep {e_t} IRR1 {res['test_top1'][e_t].sum():+.3f}"
            if len(res["val_loss"]):
                e_v = int(np.argmin(res["val_loss"]))
                msg += f" | val-selected ep {e_v} IRR1 {res['test_top1'][e_v].sum():+.3f}"
            print(msg + f" | last IRR1 {res['test_top1'][-1].sum():+.3f}", flush=True)


if __name__ == "__main__":
    main()
