"""Append manual RQ2 codes (JSON lines on stdin: {"i":NN, ...fields}) into rq2_coding.csv."""
import csv, json, sys, pathlib
p = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch" / "rq2_coding.csv"
rows = list(csv.DictReader(open(p))); F = list(rows[0].keys())
for line in sys.stdin:
    if not line.strip(): continue
    d = json.loads(line); i = d.pop("i")
    for k, v in d.items():
        assert k in F, k
        rows[i][k] = v
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, F); w.writeheader(); w.writerows(rows)
print("coded", sum(bool(r["fulltext_source"]) for r in rows), "of", len(rows))
