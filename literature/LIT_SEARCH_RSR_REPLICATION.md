# Targeted lit search — prior re-tests of RSR / relation controls in stock GNNs (2026-11-10)

Web search only. Not systematic: Scholar, SSRN, ICAIF and KDD proceedings were not searched exhaustively.

## Direct overlap with the RSR re-test
- **jinglangqiu-hub/rsr-reproduction** (GitHub, undergraduate reproduction, v0.1.0, 2026, not peer
  reviewed). https://github.com/jinglangqiu-hub/rsr-reproduction
  - Covers RSR_I and RSR_E with industry and Wiki relations on NASDAQ and NYSE.
  - Protocol A uses the author seed and the supplied embeddings. Protocol B retrains both stages
    with 5 new seeds.
  - Reproduction is partial: test-MSE deviation up to 3.23%. NASDAQ-Wiki RSR_I IRR is 0.96 in
    one run, against 1.19 in the paper.
  - Results are highly seed- and date-sensitive. NYSE-industry IRR is 1.09 ± 0.31, and −0.17 after
    excluding two dates on which every run picked the same stock.
  - **It has no Rank_LSTM / no-relation comparison, no random or placebo relations, and no
    significance tests.**
  - Relevance: it supports our motivation (seed and date fragility). It overlaps partly with stage A
    (multi-seed), and not at all with D1 inference or with stage B (placebos). We must cite it.

## Related, not overlapping
- GAPNet (arXiv 2602.00888): a learned graph with random initialisation performs only slightly
  worse than a predefined graph. NASDAQ. This is about a learned graph, not a placebo test.
- Bitcoin fraud, Elliptic (arXiv 2604.19514): randomly shuffled edges beat the real graph by 8.9
  F1. Different domain, same idea.
- Graph-augmented LSTM (arXiv 2503.03729): a retrained random-graph control in traffic and anomaly
  data, where the random graph performs on par with the no-graph model. Not finance.
- Nikolopoulos, Spurious Predictability in Financial ML (arXiv 2604.15531): a falsification audit
  with null environments and microstructure placebos. General financial ML, not graphs or relations.
- Financial ML evaluation practice: a systematic review finds significance tests are rare. One
  benchmark (arXiv 2603.01820) runs seed-robustness checks. Another (arXiv 2603.16886) finds seed
  accounts for less than 0.1% of variance, which contrasts with RSR.
- HIST, MDGNN and other CSI300 papers ablate modules or relation types. None uses shuffled or
  placebo relations.
- IJCAI 2024, "Automatic De-Biased Temporal-Relational Modeling": treats relational graphs as
  possibly erroneous. Tangential.

## Verdict
No stock-ranking paper found with retrained shuffled, permuted or placebo relation controls, or a
calibrated test of the relation claim. Stage B (within-industry permutation placebos on RSR) looks
novel. Stage A (multi-seed re-test) overlaps partly with the rsr-reproduction repo, but adds the
no-relation comparison, 20 seeds, and an A/A-calibrated test.

# Round 2 (2026-11-10 midday): gap check plus candidate models for a second re-test

Searched: arXiv, SSRN (site search), GitHub, ACM DL, and general web. Still not exhaustive. Google
Scholar was not queried directly, and ICAIF/KDD proceedings were only covered through web search.

## Gap check
- No stock-GNN paper with retrained placebo, shuffled or permuted relation controls was found in
  this round either.
- Closest neighbours:
  - Rewiring evaluation in general GNNs (arXiv 2305.19717): one-sided t-tests show rewiring rarely
    helps significantly.
  - Network spillover state-space models (arXiv 2512.18584): an exact test of whether the network
    matters, on count data (crime, COVID). Not finance prediction, not GNN.
  - Trading GNN (arXiv 2504.07923): ER random graphs inside a structural estimation, not as a
    control.
  - "Firms That Look Alike" (SSRN 7661378, Oct 2026): characteristic graphs versus a plain NN for
    US returns give mixed, specification-dependent gains. No placebo is used.
  - Sector-ETF GNN (SSRN 5976875): small margin over non-graph baselines.
  - HATS (arXiv 1908.07999): "using relation data does not always yield good results".
  - DishFT-GNN (arXiv 2502.10776): "improvements remain limited".
- Surveys: ACM CSUR systematic review of GNN stock forecasting (doi 10.1145/3696411), which lists
  RT-GCN and STHAN-SR as Wiki-relation models on RSR data.

## Candidate second re-test: RT-GCN (Zheng, Shao, Zhu, Shen, ICDE 2023)
- Code: https://github.com/zhengzetao/RTGCN. PyTorch with torchlight. Data and pretrained weights
  are in a Google Drive folder. It uses an RSR-style NASDAQ/NYSE relation tensor
  (`NASDAQ_relation.npy`).
- **Evaluation issues visible in the public code (`processor/recognition.py`), verified 2026-11-10:**
  1. There is no validation split. Only train and test feeders exist. The "best" epoch is chosen by
     **test** mean loss, and the reported MAE, MRR and top-k return come from that epoch, so model
     selection happens on the test set.
  2. Test day index 89 is deleted with `np.delete(..., 89)`, commented "in NYSY dataset, the 89-th
     day has an waivness influence TOP-n metrixc". The deletion is applied to both markets.
  These claims are about the public code only. The paper text still has to be checked to see
  whether it describes the same protocol.
- Practical issues: `.cuda()` is hard-coded, the config sets `device: [4,0]`, `num_epoch` is 100,
  and the paper's IRR numbers have not been retrieved yet. STHAN-SR has no official code; StockODE
  and others were not checked.

## RT-GCN data check (2026-11-10 ~13:15)
- The Google Drive data folder contains `data/{CSI,NASDAQ,NYSE}` and `work_dir`:
  - NYSE has `NYSE_industry_relation.npy`, `NYSE_relation.npy`, `NYSE_wiki_relation.npy`, and
    `{train,test}_15_{EOD.npy,label.pkl,price.npy}`.
  - NASDAQ has `NASDAQ_relation.npy` plus the same train/test files.
  - There are **no validation files and no ticker lists**.
- Downloaded to `data/raw/rtgcn_dl/`: only the price arrays, read with allow_pickle=False.
  - NASDAQ test (172, 854), train (1295, 854).
  - NYSE test (172, 1405).
  - Prices are normalised to max 1.
- **So RT-GCN does NOT use the RSR dataset.** Its universe is 854/1405 stocks against RSR's
  1026/1737, and its test period is 172 days, reportedly 2020-03-02 to 2020-12-31 according to the
  ACM CSUR review. Its relations carry RSR-like file names, but overlap with the RSR relations cannot
  be checked without tickers.
- Paper: IEEE Xplore doc 10184655, DOI 10.1109/ICDE55515.2023.00017 (19 citations per Crossref).
  It is paywalled and has not been read. Secondary sources (IJCAI 2024) quote RT-GCN IRR 0.48
  (NASDAQ) and 0.37 (NYSE).
- Implication: RT-GCN is a second dataset and period (2020) rather than the same benchmark. It
  allows measuring the inflation caused by test-set epoch selection. The PDF must be read before any
  claim about its protocol.

## RT-GCN paper read (2026-11-10 ~13:30)
Source: `literature/zheng2023_rtgcn_icde.pdf`, with text in `zheng2023_rtgcn.txt`. Zheng, Shao, Zhu &
Shen, ICDE 2023, pp. 123–136. The PDF carries a third-party institutional licence watermark.
**Local use only; do not redistribute it or include it in any public package.**

- **Data:** NASDAQ 854, NYSE 1405 and CSI 242 stocks. The NASDAQ/NYSE stock lists are from RSR
  [9]. Prices come from yfinance. Industry relations are filtered from the RSR data. Wiki
  relations come from the wikimedia API and were "collected before the test period".
  - Train: 2015-01-02 – 2020-02-28 (1295 days).
  - Test: 2020-03-02 – 2020-12-31 (207 days).
- **Split (§V-A, step 4):** "we chronologically split the sequential price data into two time
  periods for training and testing". **There is no validation period.**
- **Tuning (§V-B4):** grid search over window T ∈ {5, 10, 15, 20} and α ∈ {0.01, 0.1, 0.2}
  "to select the optimal hyperparameters on all graph-based methods". With no validation set, the
  selection data must be train or test. The released code selects epochs by **test** loss.
- **Runs and tests:** "we run all models fifteen times and average".
  - Table IV uses a paired Wilcoxon over 15 pairs, RT-GCN(T) against the strongest baseline. The
    p-values look like exact n = 15 values, e.g. 3.05e-5 = 1/2^15.
  - Table V uses the RSR data (NASDAQ-II / NYSE-II, industry relations only). It compares 15 RT-GCN
    runs against baseline numbers **"taken directly from [9], [10]"**, using a **one-sample
    Wilcoxon against the baseline's single reported value**. That ignores the baseline's own
    run-to-run variance.
- **Results:**
  - Table IV, IRR-1: NASDAQ RT-GCN(T) 1.25 vs RSR_E 0.89, RSR_I 0.81, Rank_LSTM 0.35. NYSE 0.92 vs
    RSR_E 0.88, RSR_I 0.66, Rank_LSTM 0.30.
  - Table V, RSR data: IRR-5 is NASDAQ 0.48 vs STHAN-SR 0.44, and NYSE 0.37 vs 0.33.
- The released config uses `edge_importance_weighting: 'Uniform'`, which is RT-GCN(U). The best
  reported variant is (T), "Time-aware".
- **Re-test angles:**
  1. Test-set selection inflation: compare selection on a validation split, on the test split,
     and at a fixed epoch, over many seeds.
  2. Matched no-relation baseline and placebo relations for RT-GCN.
  3. Running on the RSR data (Table V setting) places it on the same benchmark as our RSR re-test.

## ESWA analogue search (2026-11-10 ~19:20)
- I did not find an ESWA paper of the same type: a reproducibility study or audit of relational gains in stock GNNs.
- Closest ESWA items:
  - Jiang (2021), ESWA 184:115537. A review that emphasises implementation and reproducibility.
  - Ballings et al. (2015), ESWA. A multi-classifier comparison for stock direction.
  - Porras et al. (2022), ESWA. A literature review. It reports ESWA as the top venue for stock-forecasting papers.
- **New non-peer-reviewed threat:** GitHub `mfzhang/20260825-gnn-stocks`, mirrored as `FatihHekim0glu/gnn-stocks` (2026).
  - It compares GCN and GraphSAGE against ridge and momentum, using point-in-time correlation graphs, purged walk-forward evaluation, and Deflated Sharpe / Diebold-Mariano tests.
  - It reports an "honest null": GNNs do not beat the baselines out of sample.
  - It has no placebo graphs and does not retest a published claim, as far as I can see.
  - **It must be cited and distinguished.**
- arXiv 2510.10775, "Structure Over Signal" (OmniGNN). Despite the title, it is a new model, not an audit.
