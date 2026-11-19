"""Merge OpenAlex + arXiv RQ1 records, dedup, and flag candidates for manual title/abstract screening.
The flag is a liberal screening aid (amendment v1.1 of the protocol); flagged records are read in full,
unflagged records with a domain match are title-screened, records without an abstract are title-screened."""
import json, re, pathlib, csv
D = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch"
pool = json.load(open(D / "pool_openalex.json"))["works"]
recs = {}
def norm(t): return re.sub(r"\W+", "", (t or "").lower())
for w in pool:
    recs[norm(w["title"])] = {"src": "openalex:" + "|".join(w["sources"]), "id": w["doi"] or w["id"], "title": w["title"],
                              "year": w["year"], "venue": w["venue"] or "", "cited_by": w["cited_by"], "abstract": w["abstract"]}
for r in json.load(open(D / "arxiv_RQ1_AxBxC.json"))["results"]:
    k = norm(r["title"])
    if k in recs: recs[k]["src"] += "|arxiv_RQ1"
    else: recs[k] = {"src": "arxiv_RQ1", "id": r["id"], "title": r["title"], "year": r["published"][:4], "venue": "arXiv",
                     "cited_by": "", "abstract": r["abstract"]}
DOM = re.compile(r"stock|equit|financ|portfolio|asset|trading|investor|crypto|market", re.I)
CTL = re.compile(r"random(ly)?[- ](generated |initiali[sz]ed |constructed )?(graph|edge|relation|adjacen)|shuffl|permut|placebo|null model|"
                 r"without (the |any )?(graph|relation)|no[- ]relation|reproduc|replicat|re-?evaluat|revisit|do (graphs|relations) (help|matter)|"
                 r"necessity of|spurious|sanity check|graph[- ]free|non-graph|identity (graph|matrix)|empty graph", re.I)
rows = []
for r in recs.values():
    txt = (r["title"] or "") + " " + (r["abstract"] or "")
    r["has_abstract"] = bool(r["abstract"])
    r["domain"] = bool(DOM.search(txt))
    m = CTL.findall(txt) if r["domain"] else []
    r["flag"] = bool(CTL.search(txt)) and r["domain"]
    rows.append(r)
json.dump(rows, open(D / "rq1_records.json", "w"), indent=1)
n = len(rows); print("records", n, "domain", sum(r["domain"] for r in rows), "flagged", sum(r["flag"] for r in rows),
      "no-abstract", sum(not r["has_abstract"] for r in rows), "no-abs&domain-unknown", sum((not r["has_abstract"]) and not r["domain"] for r in rows))
