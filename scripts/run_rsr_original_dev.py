"""Run the ORIGINAL RSR TF code (data/raw/rsr_repo/training/relation_rank_lstm.py) in dev mode, for the
fidelity comparison with dapilot/rsr_replica.py (PREREG_RSR_REPLICATION.md §10).

- tensorflow -> tf.compat.v1 (TF 2.15, arm64). The training source is read and exec'd unchanged except
  that the literal seed 123456789 is replaced by --seed (the original hard-codes it in 3 places).
- Dev mode: all arrays are cut at index 1008 (no 2017 row) and valid/test indices are shifted to 504/756,
  i.e. exactly rsr_replica.Data(mode="dev").
Usage: python scripts/run_rsr_original_dev.py --market NASDAQ --seed 123456789 --out outputs/rsr_replica_dev/orig
"""
import argparse, json, os, sys, types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/rsr_repo/training"

import tensorflow as _tf_real  # noqa: E402,F401  (load the real package first)
import tensorflow.compat.v1 as tf1  # noqa: E402
tf1.disable_v2_behavior()
sys.modules["tensorflow"] = tf1

ARGS = {"NASDAQ": dict(seq=16, unit=64, alpha=0.1, emb="NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy"),
        "NYSE": dict(seq=8, unit=32, alpha=10.0, emb="NYSE_rank_lstm_seq-8_unit-32_0.csv.npy")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="NASDAQ")
    ap.add_argument("--seed", type=int, default=123456789)
    ap.add_argument("--relation", default="wikidata")
    ap.add_argument("--out", default=str(ROOT / "outputs/rsr_replica_dev/orig"))
    a = ap.parse_args()
    sys.path.insert(0, str(TRAIN))
    src = (TRAIN / "relation_rank_lstm.py").read_text()
    assert src.count("123456789") == 3
    src = src.replace("123456789", str(a.seed)).split("if __name__ == '__main__':")[0]
    mod = types.ModuleType("rrl_orig")
    os.chdir(TRAIN)
    exec(compile(src, str(TRAIN / "relation_rank_lstm.py"), "exec"), mod.__dict__)
    c = ARGS[a.market]
    m = mod.ReRaLSTM(data_path="../data/2013-01-01", market_name=a.market,
                     tickers_fname=f"{a.market}_tickers_qualify_dr-0.98_min-5_smooth.csv",
                     relation_name=a.relation, emb_fname=c["emb"],
                     parameters={"seq": c["seq"], "unit": c["unit"], "lr": 0.001, "alpha": c["alpha"]},
                     steps=1, epochs=50, batch_size=None, gpu=False, in_pro=False)
    # dev truncation: drop every row >= 1008 BEFORE training
    for k in ("eod_data", "mask_data", "gt_data", "price_data"):
        setattr(m, k, getattr(m, k)[:, :1008])
    m.embedding = np.asarray(m.embedding[:, :1008])
    m.valid_index, m.test_index, m.trade_dates = 504, 756, 1008
    vp, vg, vm, tp, tg, tm = m.train()
    ev = mod.evaluate
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"market": a.market, "seed": a.seed, "relation": a.relation,
           "val": ev(vp, vg, vm), "test": ev(tp, tg, tm)}
    res["test"]["irr_sum"] = res["test"]["btl"] - 1.0
    np.savez(out / f"{a.market}_{a.relation}_orig_s{a.seed}.npz", tp=tp, tg=tg, tm=tm)
    (out / f"{a.market}_{a.relation}_orig_s{a.seed}.json").write_text(json.dumps(res, indent=1, default=float))
    print("RESULT", json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
