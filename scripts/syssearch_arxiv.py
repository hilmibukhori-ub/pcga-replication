"""Systematic search, arXiv part (protocol: literature/SYSTEMATIC_SEARCH_PROTOCOL.md). Stdlib only."""
import json, time, urllib.parse, urllib.request, pathlib, datetime, xml.etree.ElementTree as ET

OUT = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch"
NS = {"a": "http://www.w3.org/2005/Atom"}

A = '(abs:stock OR abs:equity OR abs:"financial market" OR abs:portfolio)'
B = '(abs:"graph neural network" OR abs:GNN OR abs:"graph convolutional" OR abs:relational OR abs:hypergraph OR abs:"stock relation")'
C = ('(abs:"random graph" OR abs:shuffled OR abs:permuted OR abs:permutation OR abs:placebo OR abs:"null model" '
     'OR abs:"without graph" OR abs:reproducibility OR abs:replication OR abs:"re-evaluation")')
QUERIES = {"RQ1_AxBxC": f"{A} AND {B} AND {C}", "RQ2_AxB": f"{A} AND {B}"}


def fetch(q, start, n=200):
    u = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": start, "max_results": n, "sortBy": "submittedDate", "sortOrder": "descending"})
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "pcgv-syssearch/1.0"}), timeout=120) as r:
        return ET.fromstring(r.read())


def main():
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    for name, q in QUERIES.items():
        rows, start, total = [], 0, None
        while True:
            root = fetch(q, start)
            total = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
            ents = root.findall("a:entry", NS)
            for e in ents:
                rows.append({"id": e.find("a:id", NS).text, "title": " ".join(e.find("a:title", NS).text.split()),
                             "abstract": " ".join(e.find("a:summary", NS).text.split()),
                             "published": e.find("a:published", NS).text[:10],
                             "categories": [c.get("term") for c in e.findall("a:category", NS)]})
            start += len(ents)
            if not ents or start >= total or start >= 3000:
                break
            time.sleep(3.5)  # arXiv API etiquette
        (OUT / f"arxiv_{name}.json").write_text(json.dumps({"retrieved": stamp, "query": q, "total": total, "results": rows}, indent=1))
        print(name, "total", total, "retrieved", len(rows), flush=True)
        time.sleep(3.5)


if __name__ == "__main__":
    main()
