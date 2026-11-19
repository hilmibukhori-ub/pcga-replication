# Systematic literature search — results, round S1 (2026-11-10 evening)

Protocol: `literature/SYSTEMATIC_SEARCH_PROTOCOL.md` v1.0, plus amendment v1.1 below. Scripts:
`scripts/syssearch_openalex.py`, `scripts/syssearch_arxiv.py`, `scripts/syssearch_screen_prep.py`. Raw
outputs, with retrieval timestamps, are in `literature/syssearch/`. **Status: RQ1 stage 1 done; RQ1 backward
check and RQ2 coding are still pending.**

## Amendment v1.1 (made during execution, before any inclusion decision)
- The plain OpenAlex `search=` parameter is not boolean. It returned about 200k hits, and that run was
  discarded. Boolean blocks were run through the `title_and_abstract.search` filter instead.
- Paging without an API key uses page numbers, not cursors, and needs no personal identifier.
- **Screening aid.** About 1,270 records cannot all be read by hand in one pass, so title/abstract screening
  used two liberal aids before the manual decisions:
  1. A regex flag for control or audit terms (random graph, shuffle, permut, placebo, null model, without
     graph, reproduc, replicat, re-evaluat, revisit, spurious, sanity check, graph-free, identity graph),
     applied to records that match a finance-domain term.
  2. A title filter for evaluation-type words (revisit, rethink, evaluat, benchmark, empirical, robust,
     baseline, simple, survey, review, interpret, and others), applied to the ~1,000 forward-citation
     records.

  Every record caught by either aid was read by hand, as the matched context or the abstract. Records
  caught by neither were **not** read individually. This is a known sensitivity limitation (see below).

## PRISMA-style flow (RQ1)

| step | n |
|---|---|
| OpenAlex forward citations: RSR 587, Chen 2018 265, STHAN-SR 170, RT-GCN 22, Bechler-Speicher 15 | 1,059 |
| OpenAlex keyword query A∧B∧C (title+abstract, 2017–2026) | 440 |
| arXiv API keyword query A∧B∧C | 82 |
| **Identified** | 1,581 |
| After deduplication (DOI, then normalised title) | **1,270** |
| Finance-domain match (title/abstract) | 1,083 |
| Flagged by control-term regex | 427, of which 292 are "reproducibility" boilerplate (code-release statements) |
| Flagged by evaluation-type title filter (forward citations) | ~90 |
| Assessed on abstract, with full text where available | 17 |
| **Direct overlap (all RQ1 criteria met), peer-reviewed** | **0** |
| Partial overlap or grey literature (must be cited) | 4 |
| Adjacent (placebo or graph-evaluation idea, but not stock-GNN relation re-test) | 6 |

Supplementary web search (6 queries; GitHub, SSRN, OpenReview, general web) added 2 grey items, listed below.
None of them is a peer-reviewed direct overlap.

## Assessed records

### Partial overlap / grey literature (cite and distinguish)
1. **jinglangqiu-hub/rsr-reproduction** (GitHub, 2026, undergraduate). Multi-seed reproduction of RSR. It has
   no no-relation comparison, no placebos and no tests. (Known from earlier rounds.)
2. **BSIC, "Trading the Network II: What should an edge mean?"** (Bocconi Students Investment Club blog,
   11 Oct 2026). https://bsic.it/trading-the-network-ii-what-should-an-edge-mean/
   - Setup: GCN and GAT against ridge on US large caps, with correlation and lead-lag graphs.
   - **It includes random-graph controls.** A random residual lead-lag GAT reaches rank IC about 0.017,
     against 0.012 for the real graph. The authors conclude that GAT gains cannot be credited to the graph.
   - It has no statistical tests, no pre-specification, and does not re-test a published model.
   - **This is the closest idea match found. It is grey literature, and we must cite it.**
3. **mfzhang/20260825-gnn-stocks** (GitHub, 2026). GNNs do not beat ridge, using purged walk-forward
   evaluation with DM / Deflated Sharpe tests. It has no placebos and no published-claim re-test. (Known.)
4. **StockMixer** (Fan & Shen, AAAI 2024, doi 10.1609/aaai.v38i8.28681). A simple MLP that beats graph models
   on the RSR-type benchmarks. It is an "architecture is not needed" argument, not a relation control. Related
   to the STID argument.

### Adjacent (cite in related work)
5. **Wavelet p-Leader States and Directed Tail Hypergraphs in CPEC-Linked Pakistani Equities** (Fractal and
   Fractional 2026, doi 10.3390/fractalfract10090630).
   - It uses **999 matched random edge sets as a placebo** for 13 prespecified hyperedges, on 7 stocks, for
     risk (VaR/ES) rather than ranking.
   - The map's lift is 0.951 against a placebo median of 0.989 (p 0.834), so the structure is not confirmed.
   - It is not a GNN and not a re-test, but it is the closest finance paper that uses **edge placebos with
     inference**. It is also null.
6. **Supply Chain Propagation of Textual Signals** (arXiv 2606.29290, 2026). Asset pricing with
   network-propagated LLM embeddings. It reports "placebo experiments" and positive network predictability.
   Not a GNN and not a re-test. Its placebo design still has to be checked in the full text.
7. **Niu et al., "Evaluating Financial Relational Graphs: Interpretation Before Prediction"** (ICAIF 2024,
   doi 10.1145/3677052.3698644). Argues that judging graphs by downstream performance "conflates" graph
   quality with task fit. It has no random or no-graph controls, no seeds and no tests (full text checked).
8. **Niu et al., NGAT** (arXiv 2507.02018, 2025). The same group. Shows that the GAT-vs-NGAT ranking of graph
   constructions disagrees. Five-run averages, no tests, no placebo (full text checked).
9. **Bechler-Speicher et al., ICML 2024** (W4386620370). General GNNs overfit to graphs. All 15 of its
   OpenAlex citers were screened, and none is finance.
10. **Systematic reviews:** the ACM CSUR systematic review of GNN stock forecasting (2024); "AI/ML for Stock
    Market Prediction: SLR of … Reproducibility, and Evaluation Gaps (2016–2026)" (Research Square 2026,
    doi 10.21203/rs.3.rs-11084191/v1, PRISMA, 685 studies, which audits code and data availability). **Use
    the latter for RQ2 motivation.** Its graph-specific findings still have to be read.

### Screened out after assessment (reason)
- *Topology as a First-Class Object* (CIKM 2026): shuffling is applied to its own learned topology
  correction, as an internal ablation.
- *Entropy Injection in Fused Market Networks* (IEEE Access 2026): permutation counterfactuals for network
  structure, not prediction.
- FinBench (KDD 2026) and QuantBench (2025): benchmarks with no relation controls in the abstract. QuantBench
  names "improved methods for modelling relational data" as an open need. FinBench's full text should still
  be checked for graph ablations.
- *Leakage-Audited Sharpe Prediction* (2026): simulator study, not relational.

## Verdict for RQ1 (round S1)

- **No peer-reviewed study was found that re-tests a published stock-GNN relation claim against a matched
  no-relation baseline and retrained placebo relations with a calibrated test.**
- The idea of a random-graph control in stock GNNs exists in grey literature (BSIC blog, Oct 2026, untested).
  Edge placebos with inference exist in one adjacent finance paper (Fractal Fract 2026, risk, 7 stocks, not a
  GNN).
- Our novelty claim should be phrased as: "to our knowledge, the first **pre-registered, placebo-controlled
  re-test of published stock-GNN relation claims**". Avoid "first to use random graphs".

## Limitations of this round

- Records not caught by either screening aid were not read individually. Papers whose abstracts do not
  mention controls would only be found through full-text search or citation chasing.
- 268 records have no abstract in OpenAlex (many IEEE/Elsevier items), and these were title-screened only.
- Google Scholar, Scopus and Web of Science were not queried directly. Scopus and WoS need institutional
  access, which may be available through BCAM.
- Backward reference check of the two systematic reviews is still pending.

## Next steps
1. Backward check: reference lists of the CSUR review and the 2026 PRISMA SLR.
2. If BCAM access is available: Scopus or WoS with the same A∧B∧C block, as a sensitivity check of
   round S1.
3. RQ2 coding of the RSR forward-citation pool (protocol §RQ2 sampling).

## Sensitivity source: Scopus (2026-11-10 ~19:55, run by the user via Universitas Brawijaya access)

- Query: the protocol's A∧B∧C block as `TITLE-ABS-KEY(...) AND PUBYEAR > 2016 AND PUBYEAR < 2027`. The export
  is `syssearch/scopus_RQ1_AxBxC.csv` (original file name `export_e343773f-…T195440.csv`, same bytes), with
  44 records, all with abstracts.
- **Overlap with the S1 pool:** 23 of 44 were already in the pool, matched by DOI or title. 21 are new.
- **Screening of all 44 by hand:**
  - 0 included.
  - Already-known records reappear: NGAT, Niu et al. ICAIF 2024, Fractal Fract 2026.
  - Of the 21 new records, 18 are off-domain. The matches come from "equity" in the health-equity sense,
    "relational", or "replication".
  - Three new records are proceedings-volume entries (e-Science 2024, ACISP 2026, ACM ICPS 2018), and none of
    them contains a relevant paper title.
  - "Hybrid DL Models for Stock Market Analysis" (IEEE 2025) is a survey with no relational controls.
  - "Impact of media on Bitcoin" (Bank i Kredyt 2026) uses placebo comparisons, but it is not about graphs.
  - Also present: an ESWA 2026 attributed-multiplex GNN, which was already in the pool. Its "reproducibility"
    match is boilerplate. It goes into the RQ2 pool.
- **Conclusion:** Scopus adds no RQ1 candidate, which supports the S1 verdict. Scopus's narrower coverage of
  2025–26 conference and arXiv work makes it a weaker source than OpenAlex for this topic. The export count
  (44) should be confirmed against the hit count shown in Scopus, to make sure it is complete.

## Backward reference check (protocol source 5; 2026-11-10 ~20:15)

Script: `scripts/syssearch_backward.py`. Outputs: `syssearch/backward_CSUR2024_GNNstock.json` and
`syssearch/backward_SLR2026_PRISMA.json`. Reference lists come from OpenAlex.

| review | refs | resolved | not already in S1 pool | RQ1 includes |
|---|---|---|---|---|
| ACM CSUR 2024, "A Systematic Review on GNN-based Methods for Stock Market Forecasting" (W4402619928) | 159 | 157 | 127 | 0 |
| Research Square 2026, PRISMA SLR, "AI/ML for Stock Market Prediction … Reproducibility, and Evaluation Gaps" (W7221747120) | 215 | 212 | 203 | 0 |

- All 330 references that were not already in the pool were title-screened by hand.
- They are foundational ML/GNN works, classical econometrics, single-model stock papers, and other
  surveys. **None is a re-test or a relation-control study.**
- The PRISMA SLR's references are almost entirely non-graph stock-ML papers. Its graph coverage is thin,
  so it supports the general reproducibility motivation and says little about relational models.
- **Useful for the paper's motivation, but not RQ1:**
  - "Benchmarking Graph Neural Networks" (2020)
  - "Deep learning in the stock market — a systematic survey of practice, backtesting, and applications" (2022)
  - "Stock market prediction using artificial intelligence: A systematic review of systematic reviews" (2024)
  - "Analyzing the critical steps in deep learning-based stock forecasting: a literature review" (2024)
  - Makridakis et al., "Statistical, machine learning and deep learning forecasting methods: comparisons and ways forward" (2022)
- **Added to the RQ2 candidate pool** (relational stock models): CSUR refs "Modeling the Stock Relation with
  Graph Network for Overnight Stock Movement Prediction" (2020), ST-Trader (2021), GCNET (2022), "Forecasting
  Stock Prices Using Stock Correlation Graph: A GCN Approach" (2021), "Graph-based stock correlation and
  prediction for HFT" (2021), "A GNN-based stock forecasting method utilizing multi-source heterogeneous data"
  (2022), "Multilayer Graph-Based Deep Learning Approach for Stock Price Prediction" (2022), and
  "Financial Stock Classification Using GCN" (2023). SLR refs "The GCN Framework for Predicting Pandemic
  Impact on Stock Prices" (2022) and "Research on Financial Time Series Forecasting Algorithm Based on GCN"
  (2026).

**Updated RQ1 verdict (S1 + Scopus + backward):** there are still 0 peer-reviewed direct overlaps. The final
pool for RQ1 is 1,270 records from S1, 21 new from Scopus, and 330 new from the backward check, for
**1,621 unique records screened**.

## RQ2 coding — interim (2026-11-10 ~21:00)

- **Pool and sample.** Of the 587 RSR forward citations, 305 passed the relational and stock prefilter, and
  248 were eligible after manual title/abstract screening. Every exclusion reason is recorded in
  `syssearch/rq2_pool_screened.json`. The sample is the 30 most-cited papers plus 30 drawn at random with
  seed 20261110 (`scripts/syssearch_rq2_sample.py`). The coding sheet is `syssearch/rq2_coding.csv`.
- **Full-text access.** 19 of 60 papers are coded from open full text: arXiv, AAAI, ACL, IJCAI, IEEE
  Access open access, Research Square, and Europe PMC. The other 41 are paywalled, or the publisher blocks
  automated access. Their links are in `rq2_fulltext/TO_DOWNLOAD.md`, to be fetched by the user through
  institutional access.
- **Interim tally (n = 19; not representative yet, because 15 of the 19 are top-cited):**
  - **Random, shuffled or placebo graph control: 0 of 19.** FinGAT uses a fully-connected graph, but as
    its model rather than as a control.
  - Same-architecture ablation without the graph: about 11 of 19 clean, plus 5 partial.
  - Statistical significance test: 6 of 19. These are Wilcoxon or t-tests against baselines, mostly against
    single reported numbers.
  - Multiple runs: 13 of 19 report 5 to 10 runs, usually as a mean only.
  - Validation split used for selection: 12 of 19. **RankGLU (arXiv 2026) explicitly chooses the epoch by
    highest TEST IC, "consistent with the original protocol".** The same practice appears in the RT-GCN
    code.
  - Code released: 7 of 19.
- **New RQ1-adjacent items found during coding:**
  - Matsunaga et al. (NeurIPS 2019 workshop) re-apply an RSR-style temporal graph convolution to the Nikkei
    with rolling windows, and call RSR's relation-type finding "inconclusive" because of its short test
    period. They use no placebo and no test. It is a re-application, not a re-test.
  - RankGLU (2026) reports that removing relation-path components "does not reduce the multi-seed mean".

## RQ2 coding — final (2026-11-10 ~22:30)

**Full-text access.** 36 of 60 sampled papers were coded from full text:
- 19 from open-access PDFs or Europe PMC.
- 17 from publisher HTML read in the user's institutional Chrome session (ScienceDirect, Springer, MDPI,
  IEEE Access OA). Only targeted snippets were extracted: experimental setup, baselines, ablation, runs,
  validation, tests and code.

24 papers were coded from the abstract only and marked "not assessable":
- 11 IEEE papers behind the paywall (no institutional full text in the browser)
- 10 ACM DL papers behind a Cloudflare human check, which we did not bypass
- 2 AAAI PDFs that could not be extracted
- 1 Springer chapter preview

Coding rules are in `scripts/syssearch_rq2_summary.py`. The per-paper evidence and notes are in
`syssearch/rq2_coding.csv`.

| practice (papers with full text) | top-30 cited (n = 24) | random-30 (n = 12) | all (n = 36) |
|---|---|---|---|
| no-relation baseline present | 24 | 7 | 31 (86%) |
| same-architecture ablation without graph: clean / partial | 12 / 7 | 2 / 7 | 14 / 14 |
| **random, shuffled or placebo graph control** | **0 clean, 1 partial** | **0** | **0 clean, 1 partial (3%)** |
| multiple runs reported | 18 | 3 | 21 (58%) |
| any significance test | 7 | 2 | 9 (25%) |
| validation split used for selection: yes / not reported / test-set | 13 / 10 / 0 | 4 / 6 / 1 | 17 / 16 / 1 |
| code or data released | 9 | 2 | 11 (31%) |

**Notes.**
- **The single partial random-graph case** is AFHGN-rand in Fuzzy Hypergraph Network (Information Sciences
  2022). It initialises a *learned* hypergraph randomly instead of from the industry, concept and fund
  priors. It gives one point estimate with no test. It is not a placebo for a fixed relation graph.
- **Significance tests** are always comparisons of the proposed model against baselines: Wilcoxon, t-test,
  Friedman or paired t. None tests the graph component against a matched no-graph or placebo control.
- **Copied baseline numbers.** MGAR (Information Sciences 2023, RSR benchmark) reports Rank_LSTM rows that
  are identical to RSR's Table 7. RT-GCN (ICDE 2023) explicitly copies baseline numbers.
- **Test-set model selection.** RankGLU (arXiv 2026) states it, "consistent with the original protocol".
  The RT-GCN code does the same. Another 16 of 36 papers do not report how the model or epoch was selected.
- **Caveat on representativeness.** Full-text coverage is much higher in the top-30 stratum (24/30) than in
  the random stratum (12/30). The random-stratum counts are therefore imprecise. The headline result does
  not depend on the stratum: 0 of 36 papers use a clean placebo or random-graph control.
