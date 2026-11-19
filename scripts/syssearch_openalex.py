"""Systematic search, OpenAlex part (protocol: literature/SYSTEMATIC_SEARCH_PROTOCOL.md).

Pulls (1) forward citations of the seed papers and (2) keyword-block queries, saves raw JSON with
retrieval timestamps to literature/syssearch/, and writes a deduplicated pool with abstracts.
Stdlib only. No API key, no personal identifiers are sent.
"""
import json, time, urllib.parse, urllib.request, urllib.error, pathlib, datetime, re

OUT = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.openalex.org/works"
SELECT = "id,doi,display_name,publication_year,cited_by_count,type,primary_location,abstract_inverted_index,referenced_works_count"

SEEDS = {
    "RSR_Feng2019": "W2893230400",
    "Chen2018_CIKM": "W2896309423",
    "RTGCN_Zheng2023": "W4385284998",
    "BechlerSpeicher2024": "W4386620370",
    "STHANSR_Sawhney2021": "W3175177406",
}

A = '(stock OR equity OR "financial market" OR portfolio OR "asset return")'
B = '("graph neural network" OR GNN OR "graph convolutional" OR relational OR "stock relation" OR hypergraph OR "knowledge graph")'
C = ('("random graph" OR "randomized graph" OR shuffled OR permuted OR permutation OR placebo OR "null model" '
     'OR "without graph" OR "no relation" OR reproducibility OR replication OR "re-evaluation" OR "do graphs help")')
# Boolean queries go through the title_and_abstract.search filter (the plain `search` param is not boolean).
QUERIES = {"RQ1_AxBxC": f"{A} AND {B} AND {C}"}


def get(params, tries=4):
    url = API + "?" + urllib.parse.urlencode(params)
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pcgv-syssearch/1.0"}), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            print("HTTP", e.code, body, flush=True)
            if t == tries - 1 or e.code == 401:
                raise
            time.sleep(3 * (t + 1))


def pull(filt=None, search=None, cap=2000):
    # page-number paging (cursor paging needs an API key; page paging is fine up to 10k results)
    rows, page, count = [], 1, None
    while len(rows) < cap:
        p = {"per-page": 200, "page": page, "select": SELECT}
        if filt: p["filter"] = filt
        if search: p["search"] = search
        d = get(p)
        count = d["meta"].get("count")
        rows += d["results"]
        if not d["results"] or len(rows) >= (count or 0):
            break
        page += 1
        time.sleep(1.2)  # heavy boolean queries: 1 req/s without a key
    return rows, count


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, ix in inv.items() for i in ix)
    return " ".join(w for _, w in pos)


def main():
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    pool, flow = {}, {}
    jobs = [(f"cites_{k}", {"filt": f"cites:{v}"}) for k, v in SEEDS.items()]
    jobs += [(k, {"filt": f"title_and_abstract.search:{q},publication_year:2017-2026"}) for k, q in QUERIES.items()]
    for name, kw in jobs:
        rows, count = pull(**kw)
        (OUT / f"openalex_{name}.json").write_text(json.dumps({"retrieved": stamp, "query": kw, "count": count, "results": rows}))
        flow[name] = {"count_reported": count, "retrieved": len(rows)}
        for r in rows:
            key = (r.get("doi") or "").lower() or re.sub(r"\W+", "", (r["display_name"] or "").lower())
            e = pool.setdefault(key, {"id": r["id"], "doi": r.get("doi"), "title": r["display_name"],
                                      "year": r["publication_year"], "cited_by": r["cited_by_count"],
                                      "type": r["type"],
                                      "venue": ((r.get("primary_location") or {}).get("source") or {}).get("display_name"),
                                      "abstract": abstract(r.get("abstract_inverted_index")), "sources": []})
            e["sources"].append(name)
        print(name, flow[name], flush=True)
    (OUT / "pool_openalex.json").write_text(json.dumps({"retrieved": stamp, "flow": flow, "n_dedup": len(pool), "works": list(pool.values())}, indent=1))
    print("dedup pool", len(pool))


if __name__ == "__main__":
    main()
