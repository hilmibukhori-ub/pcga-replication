# Systematic literature search — protocol (v1.0, fixed 2026-11-10 before execution)

PRISMA-style, adapted for a methods/novelty search. Executed by script so every step can be rerun.
Raw outputs: `literature/syssearch/` (one JSON per source/query, with retrieval timestamp).

## Research questions

- **RQ1 (novelty).** Has any study tested whether the relation graph of a stock-prediction GNN adds value
  by (a) re-running a published model against a matched no-relation baseline and/or (b) retraining with
  placebo, random, shuffled or permuted relations, with a statistical test?
- **RQ2 (practice survey, feeds a methods table).** In relational stock-prediction papers, how is the
  contribution of the relation graph evaluated: no-relation baseline, random/placebo graph, number of
  runs/seeds, significance test, validation split for model selection?

## Sources

1. **OpenAlex API** (covers ACM DL, IEEE, Elsevier/ESWA, Springer, SSRN, arXiv). Keyword queries below,
   title+abstract search, publication years 2017–2026.
2. **arXiv API** (q-fin, cs.LG, cs.AI, cs.CE), same keyword blocks.
3. **Forward citation chasing (OpenAlex `cites:`):** every work citing
   - Feng et al. 2019 TOIS "Temporal Relational Ranking for Stock Prediction" (RSR)
   - Chen et al. 2018 CIKM "Incorporating corporation relationship via graph convolutional neural networks for stock price prediction"
   - Zheng et al. 2023 ICDE RT-GCN
   - Bechler-Speicher et al. 2024 ICML "Graph Neural Networks Use Graphs When They Shouldn't" (nearest general-ML threat)
   - Sawhney et al. 2021 AAAI STHAN-SR (major RSR-benchmark successor)
4. **Supplementary web search** (Google-Scholar substitute; GitHub, SSRN, OpenReview) with the same
   key phrases. Not counted as a database; reported separately.
5. **Backward check:** reference lists of the two systematic reviews of GNN stock forecasting found in
   rounds 1–2 (ACM CSUR doi 10.1145/3696411; any other review found).

## Query blocks (title/abstract)

- **A (domain):** stock OR equity OR "financial market" OR portfolio OR "asset return"
- **B (relational model):** "graph neural network" OR GNN OR "graph convolutional" OR "relational" OR
  "stock relation" OR hypergraph OR "knowledge graph"
- **C (control / audit):** random graph OR "randomized graph" OR shuffled OR permuted OR permutation OR
  placebo OR "null model" OR "without graph" OR "no relation" OR reproducibility OR replication OR
  "re-evaluation" OR "do graphs help" OR "graph is necessary"

RQ1 query = A AND B AND C. RQ2 candidate pool = A AND B (plus forward citations of RSR), sampled as in
"RQ2 sampling".

## Eligibility

**RQ1 include:** stock/equity prediction, ranking or portfolio task; relation graph (predefined or
learned); and either a re-run of a published relational model, or a retrained control graph
(random/shuffled/permuted/placebo/degree-preserving), or an explicit no-graph comparison of the *same*
architecture with inference. Any language of publication is screened only if an English abstract exists.
Preprints, theses and code repositories are eligible (grey literature, flagged).
**RQ1 exclude:** non-financial domains (kept separately as "adjacent", e.g. traffic, fraud, general
GNN benchmarks); ablations that only remove modules without a control graph.

**RQ2 include:** peer-reviewed or arXiv paper proposing or evaluating a relational stock-prediction
model with an empirical evaluation on stock data.

## Screening

Two stages: (1) title+abstract, (2) full text for every RQ1 candidate.

## RQ2 sampling and coding

Pool: forward citations of RSR that pass RQ2 eligibility. Code all of them if ≤ 60, otherwise the 30
most cited plus a uniform random sample of 30 (seed 20261110). Coding sheet fields:
no-relation baseline (same architecture?), random/placebo graph, #runs, significance test (which),
validation split used for model/epoch selection, code released, dataset.

## Reporting

PRISMA flow counts (identified, deduplicated, screened, full-text, included) per RQ, in
`literature/SYSTEMATIC_SEARCH_RESULTS.md`.
