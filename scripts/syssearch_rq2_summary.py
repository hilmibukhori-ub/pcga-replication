"""Summarise RQ2 coding sheet into categorical counts per stratum (rules documented inline)."""
import csv, re, pathlib, collections
p = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch" / "rq2_coding.csv"
rows = list(csv.DictReader(open(p)))
def cat(field, v):
    v = (v or "").lower()
    if "not assessable" in v: return "NA"
    if field == "random_or_placebo_graph":
        return "partial" if "partial" in v else ("yes" if v.startswith("yes") else "no")
    if field == "sig_test":
        return "no" if (v.startswith("none") or "correlation test only" in v or v == "") else "yes"
    if field == "n_runs":
        if re.search(r"\b(5|10|five|ten|several|multiple|repeat)", v) and "not reported" not in v: return "yes"
        return "no"
    if field == "validation_split_for_selection":
        if v.startswith("no:") or "test-set" in v or "test ic" in v: return "test-set selection"
        if v.startswith("yes"): return "yes"
        if v.startswith("partial"): return "partial"
        return "not reported"
    if field in ("same_arch_no_graph", "no_relation_baseline"):
        if v.startswith("yes"): return "yes"
        if v.startswith("partial"): return "partial"
        return "no/not reported"
    if field == "code_released":
        return "yes" if (v.startswith("yes") or "promised" in v or "data released" in v) else "no"
FIELDS = ["no_relation_baseline", "same_arch_no_graph", "random_or_placebo_graph", "n_runs", "sig_test", "validation_split_for_selection", "code_released"]
out = []
for strat in ["top30", "random30", "all"]:
    R = [r for r in rows if strat == "all" or r["stratum"] == strat]
    A = [r for r in R if not r["fulltext_source"].startswith("abstract")]
    out.append(f"\n### {strat}: n={len(R)}, full text assessed={len(A)}")
    for f in FIELDS:
        c = collections.Counter(cat(f, r[f]) for r in A)
        out.append(f"- {f}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
print("\n".join(out))
