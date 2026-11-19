"""RSR intake checks (PREREG_RSR.md §2.3). Structure only: no return, loss, or correlation statistic.

Records provenance + sha256 manifest, stock/day counts, date ranges per split, missing rates per
split, industry-group sizes, and Wiki-graph structure. Writes outputs/rsr_intake/.

    python -I scripts/rsr_intake.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/rsr_repo"
REL = ROOT / "data/raw/rsr_relation/relation"
OUT = ROOT / "outputs/rsr_intake"
PAD = 29          # eod.py pad_begin: processed row r <-> trading date index r + 29
MISSING = -1234.0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tickers(m):
    return np.genfromtxt(REPO / f"data/{m}_tickers_qualify_dr-0.98_min-5_smooth.csv", dtype=str).tolist()


def dates(m):
    d = [l.strip()[:10] for l in open(REPO / f"data/{m}_aver_line_dates.csv") if l.strip()]
    return d[PAD:]


def load_close(m, tickers):
    """Only the missing-indicator of the close column is used here (no values)."""
    cols = []
    for t in tickers:
        a = np.loadtxt(REPO / f"data/2013-01-01/{m}_{t}_1.csv", delimiter=",", dtype=np.float64)
        if m == "NASDAQ":
            a = a[:-1]          # RSR load_data.py drops the last NASDAQ day
        cols.append(np.isclose(a[:, -1], MISSING))
    return np.stack(cols, 1)    # (T, n) bool: True = missing close


def adjacency(npy):
    r = np.load(npy)
    A = r.sum(2) > 0
    return r.shape, A


def main():
    cfg = yaml.safe_load((ROOT / "configs/rsr_prereg.yaml").read_text())
    split = cfg["data"]["split"]
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rep = {"repo": cfg["data"]["source"], "commit": commit, "license": "AGPL-3.0 (code); data rights not separately stated",
           "markets": {}, "hard_stops": []}
    manifest = {}

    for m, spec in cfg["data"]["markets"].items():
        tick = read_tickers(m)
        files = [REPO / f"data/2013-01-01/{m}_{t}_1.csv" for t in tick]
        extra = [REPO / f"data/{m}_tickers_qualify_dr-0.98_min-5_smooth.csv", REPO / f"data/{m}_aver_line_dates.csv",
                 REL / f"sector_industry/{m}_industry_relation.npy", REL / f"sector_industry/{m}_industry_ticker.json",
                 REL / f"wikidata/{m}_wiki_relation.npy", REL / f"wikidata/{m}_connections.json"]
        for p in files + extra:
            manifest[str(p.relative_to(ROOT))] = sha256(p)

        d = dates(m)
        miss = load_close(m, tick)
        T, n = miss.shape
        d = d[:T]
        r = {"stocks": n, "expected_stocks": spec["expected_stocks"], "processed_days": T,
             "first_date": d[0], "last_date": d[-1]}
        if n != spec["expected_stocks"]:
            rep["hard_stops"].append(f"{m}: {n} stocks != {spec['expected_stocks']}")

        idx = {}
        for s, (a, b) in split.items():
            ii = [i for i, x in enumerate(d) if a <= x <= b]
            idx[s] = ii
            r[f"{s}_rows"] = [ii[0], ii[-1]] if ii else None
            r[f"{s}_days"] = len(ii)
            r[f"{s}_dates"] = [d[ii[0]], d[ii[-1]]] if ii else None
            r[f"{s}_missing_close_rate"] = float(miss[ii].mean()) if ii else None
            # a loss needs close at t and t+1 (§2.4)
            if ii:
                jj = [i for i in ii if i + 1 < T]
                pair_ok = ~miss[jj] & ~miss[[i + 1 for i in jj]]
                r[f"{s}_loss_mask_rate"] = float(pair_ok.mean())
                r[f"{s}_stocks_all_missing"] = int((miss[ii].all(0)).sum())
        if r["train_dates"][0] != split["train"][0] or r["test_dates"][1] != split["test"][1]:
            rep["hard_stops"].append(f"{m}: date range {r['train_dates'][0]}..{r['test_dates'][1]} != prereg")
        unassigned = T - sum(r[f"{s}_days"] for s in split)
        r["rows_outside_splits"] = unassigned

        # industry
        ind = json.loads((REL / f"sector_industry/{m}_industry_ticker.json").read_text())
        tset = set(tick)
        label = {}
        for g, members in ind.items():
            for t in members:
                if t in tset:
                    label.setdefault(t, []).append(g)
        sizes = np.array(sorted([sum(t in tset for t in mem) for mem in ind.values()], reverse=True))
        sizes = sizes[sizes > 0]
        r["industry"] = {"groups_with_members": int(len(sizes)), "stocks_labelled": len(label),
                         "stocks_multi_label": int(sum(len(v) > 1 for v in label.values())),
                         "stocks_unlabelled": n - len(label),
                         "size_max": int(sizes.max()), "size_median": float(np.median(sizes)),
                         "size_quantiles_10_50_90": np.quantile(sizes, [.1, .5, .9]).tolist(),
                         "singleton_groups": int((sizes == 1).sum())}
        shp, Ai = adjacency(REL / f"sector_industry/{m}_industry_relation.npy")
        r["industry"]["npy_shape"] = list(shp)
        r["industry"]["npy_diag_nonzero"] = int(np.diag(Ai).sum())

        # wiki
        shp, Aw = adjacency(REL / f"wikidata/{m}_wiki_relation.npy")
        np.fill_diagonal(Aw, False)
        sym = bool((Aw == Aw.T).all())
        Au = Aw | Aw.T
        deg = Au.sum(1)
        gi = {t: (label.get(t) or [None])[0] for t in tick}
        iu = np.argwhere(np.triu(Au, 1))
        same = np.array([gi[tick[a]] is not None and gi[tick[a]] == gi[tick[b]] for a, b in iu])
        r["wiki"] = {"npy_shape": list(shp), "relation_channels": shp[2],
                     "diag_nonzero_before_removal": int(np.diag(adjacency(REL / f"wikidata/{m}_wiki_relation.npy")[1]).sum()),
                     "symmetric_as_given": sym, "edges_undirected": int(len(iu)),
                     "isolated_nodes": int((deg == 0).sum()), "isolated_share": float((deg == 0).mean()),
                     "degree_mean": float(deg.mean()), "degree_max": int(deg.max()),
                     "degree_quantiles_50_90_99": np.quantile(deg, [.5, .9, .99]).tolist(),
                     "share_edges_same_industry": float(same.mean()) if len(same) else None}
        rep["markets"][m] = r

    (OUT / "manifest_sha256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    rep["manifest_sha256_of_manifest"] = sha256(OUT / "manifest_sha256.json")
    rep["n_files_hashed"] = len(manifest)
    (OUT / "intake.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["hard_stops"]:
        print("HARD STOP:", rep["hard_stops"], file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
