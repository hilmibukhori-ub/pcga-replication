# RSR intake record (PREREG_RSR.md §2.3)

**Run:** 2026-11-09, `python -I scripts/rsr_intake.py`. Machine-readable output: `intake.json`.
**Scope:** structure only. No return, loss, or correlation statistic was computed on any split.

## Provenance

- Repository: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking, commit
  `cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d` (last commit 2021-03-04), shallow clone in
  `data/raw/rsr_repo/`. `data/relation.tar.gz` is extracted to `data/raw/rsr_relation/`.
- Licence: AGPL-3.0 for the repository. The data rights are not stated separately, and the prices
  come from legacy Google Finance. Raw data are not redistributed.
- 2,775 files hashed in `manifest_sha256.json` (sha256 of the manifest:
  `a2bfd23079bd339373bb750bea5ee07dbcc4f4597d67f19211c9901704b179b4`).
- Prices used: the RSR processed files `data/2013-01-01/{market}_{ticker}_1.csv`, last column. This
  is the close divided by a per-stock constant (its max close), so returns are unaffected. A
  missing close is coded −1234, and a close is also set to missing when the previous trading day
  has no record (`eod.py`). The NASDAQ last row is dropped, as in RSR `load_data.py`.

## Hard stops: none

| | NASDAQ | NYSE |
|---|---|---|
| stocks (expected) | 1026 (1026) | 1737 (1737) |
| days, train / val / test | 756 / 252 / **237** | 756 / 252 / **237** |
| date range | 2013-01-02 → 2017-12-08 | 2013-01-02 → 2017-12-08 |
| missing close, train / val / test | 0.14% / 0.22% / 0.05% | 0.08% / 0.04% / 0.06% |

## Industry labels

| | NASDAQ | NYSE |
|---|---|---|
| groups / singleton groups | 113 / 17 | 130 / 23 |
| largest group | **`n/a`: 156** | **`n/a`: 500 (29% of stocks)** |
| median group size | 4 | 6 |

RSR's own industry relation treats `n/a` as one industry: every pair of `n/a` stocks is connected
(`preprocess/sector_industry.py`). → Clarification C2 in `DEVIATIONS.md`.

## Wiki graph (L4)

| | NASDAQ | NYSE |
|---|---|---|
| undirected edges | 970 | 4,246 |
| isolated nodes | **891 (87%)** | **1,471 (85%)** |
| mean / max degree | 1.9 / 54 | 4.9 / 113 |
| degree 50% / 90% / 99% | 0 / 3 / 38 | 0 / 10 / 85 |
| share of edges within the same industry | 13% | 11% |
| symmetric as given | no (symmetrised, C1) | no (symmetrised, C1) |

The Wiki graph is far sparser and more hub-dominated than the graph simulated in the power study
(about 6 neighbours per node, no isolated nodes). Only about 135 NASDAQ stocks and about 266 NYSE
stocks have any neighbour. The power in PREREG §7 is therefore likely to be considerably
overstated.
