"""RQ2: eligibility (manual title/abstract decisions keyed by title) + stratified sample (protocol §RQ2 sampling)."""
import json, random, csv, pathlib
D = pathlib.Path(__file__).resolve().parents[1] / "literature" / "syssearch"
c = sorted(json.load(open(D / "rq2_prepool.json")), key=lambda x: -x["cited_by"])  # same order as the screening listing
EXC_IDX = {17:'MLP, no relation graph',31:'general hypergraph method, not stock',32:'general GNN (fraud)',37:'KG trend detection, not stock prediction',
41:'alpha mining, no relation graph',53:'ESG ratings, not returns',60:'unclear/not stock relational',66:'fundraising',82:'fundraising',96:'causal discovery, no prediction model',
98:'budgeting',99:'bonds',100:'bonds',101:'survey',106:'mutual funds',107:'option pricing',110:'embeddings for asset mgmt, no prediction eval',113:'TS pretraining, no relation',
116:'retail',130:'visual analytics',133:'benchmark',144:'LLM anonymisation, no graph',145:'general TS',146:'retail traders',147:'credit risk',154:'model recommendation',
156:'MSMEs clustering',159:'product demand',160:'labeling, no graph',162:'exchange rates',178:'risk contagion',180:'risk contagion',185:'environment forecasting',
188:'index clustering',194:'credit spreads',200:'KG source evaluation, recommendation',204:'TS embedding',206:'linguistics',209:'benchmark',216:'benchmark',218:'foundation models',
225:'metal prices',232:'network analysis only',234:'synthetic data generation',236:'asset pricing factors, no graph',242:'transactions',245:'LLM',246:'domain discovery',
251:'corporate profiling',255:'network structure, no prediction',261:'review',268:'risk contagion',269:'volatility spillover multi-asset',280:'contagion',283:'diagnostics',284:'no graph',296:'alpha mining'}
EXC = {c[i]["title"]: r for i, r in EXC_IDX.items()}
for x in c:
    x["rq2_eligible"] = x["title"] not in EXC; x["exclusion_reason"] = EXC.get(x["title"], "")
json.dump(c, open(D / "rq2_pool_screened.json", "w"), indent=1)
elig = [x for x in c if x["rq2_eligible"]]
top, rest = elig[:30], elig[30:]
rnd = random.Random(20261110).sample(rest, 30)
sample = [dict(x, stratum="top30") for x in top] + [dict(x, stratum="random30") for x in rnd]
F = ['stratum','title','year','venue','cited_by','doi','id','fulltext_source','task','dataset','graph_type','no_relation_baseline',
     'same_arch_no_graph','random_or_placebo_graph','n_runs','sig_test','validation_split_for_selection','code_released','notes']
with open(D / "rq2_coding.csv", "w", newline="") as f:
    w = csv.DictWriter(f, F); w.writeheader()
    for x in sample: w.writerow({k: x.get(k, "") for k in F})
print("prepool", len(c), "eligible", len(elig), "sample", len(sample))
for i, x in enumerate(sample): print(i, x["stratum"][:3], x["year"], x["doi"] or x["id"], "|", x["title"][:75])
