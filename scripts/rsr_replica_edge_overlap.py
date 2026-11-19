"""Prereg §8.6 (descriptive, no test data): mean edge overlap between the 99 placebo relation graphs and the
real Wiki graph. Overlap = share of real non-self edges (i,j), i != j, that are also non-self edges of the placebo
(edge presence only, any type). Also the share with identical typed relation vector. Regenerates placebos
exactly as scripts/run_rsr_replica.py does."""
import sys, json, pathlib, numpy as np, yaml
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dapilot import rsr_replica as rr
cfg = yaml.safe_load(open(ROOT / "configs/rsr_replica.yaml"))
res = {}
for mi, market in enumerate(list(cfg["markets"])):
    rel = rr.load_relation(market, "wikidata")
    i, j, r = rr.to_edges(rel)
    keep = i != j
    real = {(a, b): tuple(v) for a, b, v in zip(i[keep], j[keep], r[keep])}
    labels = rr.industry_labels(market)
    ov, ovt = [], []
    for k in range(cfg["placebos"]["K"]):
        pi = rr.within_label_perm(labels, np.random.default_rng([cfg["experiment"]["seed"], mi, 7, k]))
        pi_, pj, pr = rr.permute_edges((i, j, r), pi)
        pk = pi_ != pj
        plac = {(a, b): tuple(v) for a, b, v in zip(pi_[pk], pj[pk], pr[pk])}
        ov.append(np.mean([e in plac for e in real]))
        ovt.append(np.mean([plac.get(e) == v for e, v in real.items()]))
    res[market] = {"n_real_edges": len(real), "overlap_mean": float(np.mean(ov)), "overlap_min": float(np.min(ov)),
                   "overlap_max": float(np.max(ov)), "typed_overlap_mean": float(np.mean(ovt))}
    print(market, res[market])
json.dump(res, open(ROOT / "outputs/rsr_replica_v1/edge_overlap_s8_6.json", "w"), indent=1)
