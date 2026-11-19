"""Dev rehearsal analysis (no 2017 data): seed noise, A/A calibration of candidate D1 tests, placebo
spread, and original-TF vs port comparison. Output feeds PREREG_RSR_REPLICATION.md §10."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rsr_replica as rr  # noqa: E402
from dapilot import rsr_replica_infer as inf  # noqa: E402

D = ROOT / "outputs/rsr_replica_dev"
market = sys.argv[1] if len(sys.argv) > 1 else "NASDAQ"


def load(graph):
    out = {}
    for f in sorted(D.glob(f"{market}_wikidata_{graph}_s*.npz")):
        out[int(f.stem.rsplit("_s", 1)[1])] = dict(np.load(f))
    return out


none, real = load("none"), load("real")
plac = {f.stem.split("_")[2]: dict(np.load(f)) for f in sorted(D.glob(f"{market}_wikidata_perm*_s0.npz"))}
irr = lambda m: float(m["test_top1"].sum())  # noqa: E731
mse = lambda m: float(m["test_se"].sum() / m["test_cnt"].sum())  # noqa: E731
print(f"== {market} dev (pseudo-test = 2016)")
print(f"L0 seeds {sorted(none)}: IRR " + " ".join(f"{irr(none[s]):+.2f}" for s in sorted(none)))
print(f"   mean {np.mean([irr(m) for m in none.values()]):+.3f} sd {np.std([irr(m) for m in none.values()], ddof=1):.3f}"
      f" | best epochs {[int(m['best_epoch']) for m in none.values()]}")
print(f"real seeds {sorted(real)}: IRR " + " ".join(f"{irr(real[s]):+.2f}" for s in sorted(real)))
if len(real) > 1:
    print(f"   mean {np.mean([irr(m) for m in real.values()]):+.3f} sd {np.std([irr(m) for m in real.values()], ddof=1):.3f}"
          f" | best epochs {[int(m['best_epoch']) for m in real.values()]}")
common = sorted(set(real) & set(none))
if len(common) >= 3:
    print(f"CRN corr of per-seed IRR (real vs L0, seeds {common}): "
          f"{np.corrcoef([irr(real[s]) for s in common], [irr(none[s]) for s in common])[0, 1]:+.2f}")
# day-only bootstrap SE for a single seed vs seed sd
rng = np.random.default_rng(0)
s0 = none[sorted(none)[0]]["test_top1"]
T = len(s0); L = 20
boots = []
for _ in range(2000):
    st = rng.integers(0, T, round(T / L)); idx = ((st[:, None] + np.arange(L)) % T).ravel()
    boots.append(s0[idx].mean() * T)
print(f"day-bootstrap SE of one model's IRR: {np.std(boots):.3f}  vs seed sd of IRR above")

# A/A calibration on L0 seeds
S = sorted(none)
if len(S) >= 10:
    A_all = np.stack([none[s]["test_top1"] for s in S])
    for g in (5, max(5, len(S) // 2)):
        rej = {"day_boot": [], "seed_t": [], "twoway": []}
        for k in range(200):
            perm = rng.permutation(len(S)); a, b = perm[:g], perm[g:2 * g]
            A, B = A_all[a], A_all[b]
            rej["day_boot"].append(inf.day_bootstrap(A, B, 20, 500, rng) <= 0.05)
            rej["seed_t"].append(inf.seed_t(A, B, paired=False) <= 0.05)
            rej["twoway"].append(inf.twoway_bootstrap(A, B, 20, 300, rng, paired=False) <= 0.05)
        print(f"A/A ({g} vs {g} L0 seeds, 200 random splits of {len(S)} seeds; nominal 0.05): "
              + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in rej.items()))

# placebo spread (seed 0)
if plac and 0 in real:
    P = np.array([irr(m) for m in plac.values()])
    print(f"placebos (seed 0, n={len(P)}): IRR mean {P.mean():+.3f} sd {P.std(ddof=1):.3f} min {P.min():+.2f} max {P.max():+.2f};"
          f" real s0 {irr(real[0]):+.3f} -> {int((P >= irr(real[0])).sum())}/{len(P)} placebos >= real")
    print(f"   placebos beating L0(s0) by >0.09: {(P - irr(none[0]) > 0.09).mean():.2f}")
    Pm = np.array([mse(m) for m in plac.values()])
    print(f"   MSE placebo sd {Pm.std(ddof=1):.2e}; real s0 {mse(real[0]):.4e}; L0 s0 {mse(none[0]):.4e}")

# original TF vs port
orig = sorted((D / "orig").glob(f"{market}_wikidata_orig_s*.json"))
if orig:
    print("ORIGINAL TF code (dev):")
    for f in orig:
        r = json.loads(f.read_text())
        print(f"   seed {r['seed']}: test IRR {r['test']['btl'] - 1:+.3f} mrr {r['test']['mrrt']:.4f} mse {r['test']['mse']:.4e}"
              f" | val IRR {r['val']['btl'] - 1:+.3f} mse {r['val']['mse']:.4e}")
    print("PORT (real):")
    for s, m in sorted(real.items()):
        print(f"   seed {s}: test IRR {irr(m):+.3f} mrr {m['test_rr'].mean():.4f} mse {mse(m):.4e}"
              f" | val IRR {m['val_top1'].sum():+.3f} mse {m['val_se'].sum() / m['val_cnt'].sum():.4e} | ep* {int(m['best_epoch'])}")
