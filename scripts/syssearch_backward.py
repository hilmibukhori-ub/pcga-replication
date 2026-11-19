"""Backward reference check (protocol source 5): references of the two systematic reviews, via OpenAlex.
Saves syssearch/backward_<review>.json and flags records not already in rq1_records.json."""
import json, re, time, urllib.parse, urllib.request, pathlib, datetime

D = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch"
REVIEWS = {"CSUR2024_GNNstock": "doi:10.1145/3696411", "SLR2026_PRISMA": "doi:10.21203/rs.3.rs-11084191/v1"}
SEL = "id,doi,display_name,publication_year,cited_by_count,type,primary_location,abstract_inverted_index"


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pcgv-syssearch/1.0"}), timeout=60) as r:
        return json.load(r)


def abstract(inv):
    return " ".join(w for _, w in sorted((i, w) for w, ix in (inv or {}).items() for i in ix))


norm = lambda t: re.sub(r"\W+", "", (t or "").lower())
pool = json.load(open(D / "rq1_records.json"))
known_doi = {(r["id"] or "").lower().replace("https://doi.org/", "") for r in pool}
known_t = {norm(r["title"]) for r in pool}
stamp = datetime.datetime.now().isoformat(timespec="seconds")
for name, ref in REVIEWS.items():
    w = get(f"https://api.openalex.org/works/{ref}?select=id,display_name,referenced_works")
    ids = [u.rsplit("/", 1)[1] for u in w["referenced_works"]]
    rows = []
    for k in range(0, len(ids), 50):
        q = urllib.parse.urlencode({"filter": "openalex_id:" + "|".join(ids[k:k + 50]), "per-page": 50, "select": SEL})
        rows += get("https://api.openalex.org/works?" + q)["results"]
        time.sleep(0.3)
    out = []
    for r in rows:
        doi = (r.get("doi") or "").lower().replace("https://doi.org/", "")
        out.append({"id": r["id"], "doi": doi, "title": r["display_name"], "year": r["publication_year"],
                    "cited_by": r["cited_by_count"],
                    "venue": ((r.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "abstract": abstract(r.get("abstract_inverted_index")),
                    "in_pool": (doi in known_doi and bool(doi)) or norm(r["display_name"]) in known_t})
    json.dump({"retrieved": stamp, "review": w["display_name"], "n_refs": len(ids), "works": out},
              open(D / f"backward_{name}.json", "w"), indent=1)
    print(name, "refs", len(ids), "resolved", len(out), "new (not in pool)", sum(not o["in_pool"] for o in out))
