# Pre-registration — RSR open-track graph-value ladder (v0.2)

**Status:** FROZEN 2026-11-09 (see `PREREG_FREEZE.md`). Machine-readable spec:
`configs/rsr_prereg.yaml`.
**Drafted:** 2026-11-09 (v0.1). **v0.2:** 2026-11-09. All four open decisions were resolved as
recommended (see the end of this document).
**Data status:** no RSR data has been downloaded, opened, or inspected in this workspace.
**Track:** O (open legacy replication), as defined in `BENCHMARK_FREEZE.md` §2. All results are
fixed-universe historical results and are not market-wide claims.

Upstream documents (sha256):

| File | sha256 |
|---|---|
| `BENCHMARK_FREEZE.md` | `5f8a5a465954895684c3f219c9c1f7c31834b779adc5764de8988551dd7c6846` |
| `configs/benchmark_freeze.yaml` | `b1e804cd7b1839455eeee7c56e75ea12ba7ddcda1f28801e38b7dfe7072942e7` |
| `configs/power_ladder_v2.yaml` | `554acbb1d3b808510545bfbac242234392d4b7e43f3482846ba1811caf6caf89` |
| `configs/power_ladder_v2_1.yaml` | `cf691a8c776ac32611b7f1f40c8e59a0f6b3ce37406c61b282a5a81392b772d7` |
| `outputs/power_ladder_v2/size_and_selection.csv` | `76bb8dbbc8ad541fbcf8b97777245d16b4907b351054a3c3abfe62b801e0d9f6` |

At freeze time, this document and the analysis config/code are hashed, and the hashes are
recorded in an immutable record (git tag or timestamped archive) **before** the test period is
read.

---

## 1. Questions

For each RSR market, using a matched linear graph expert:

- **Q1 (access value, D1):** Does adding neighbour aggregates from the real relation graph lower
  out-of-sample next-day loss compared with the same model with no graph?
- **Q2 (edge value, `edge` = D2 against L3):** Does the real graph lower loss compared with
  retrained placebo graphs that preserve each node's degree **and** its number of same-industry
  neighbours? Put differently, is there value in *which* specific edges exist, beyond density,
  degree, and industry composition?

The intermediate ladder steps (`density`, `degree`, `industry`) are reported descriptively. They
are not part of the confirmatory family, because the power study found them underpowered at RSR
size (`outputs/power_ladder_v1/REPORT.md`).

## 2. Data

### 2.1 Source and provenance

- RSR repository (Feng et al., *Temporal Relational Stock Ranking*), NASDAQ and NYSE subsets:
  daily price/volume, industry relations, and Wiki relations.
- Before analysis, record: repository URL, commit hash, download date, sha256 of every raw file,
  and the licence/redistribution language (AGPL-3.0 code; data rights unclear). Raw data are not
  committed. This clears the `rsr_provenance_and_redistribution_record` gate in the freeze.

### 2.2 Split (published RSR split, unchanged)

| Split | Dates | Use |
|---|---|---|
| train | 2013-01-02 → 2015-12-31 | fitting |
| validation | 2016-01-04 → 2016-12-30 | ridge α, graph hyperparameters, winsorisation check |
| test (locked) | 2017-01-03 → 2017-12-08 | confirmatory evaluation only |

The last decision date of train and of validation is purged, so that every label matures inside
its own split (the v2 maturity rule).

> The test period has about 237 trading days, not the 252 used in the power study. The power
> figures in §7 are therefore slightly optimistic. The difference is not corrected; it is reported.

### 2.3 Intake checks (allowed on all splits; no returns are examined)

At intake, only the following may be computed and recorded: number of stocks and trading days per
market, the date range, the missing-value encoding and missing rate per split, the number of
industry groups and the industry-size distribution, and Wiki relation counts, degree
distribution, and number of isolated nodes. No return, loss, or correlation statistic on the test
period is computed until the freeze.

**Hard stops:** stock counts that differ from 1,026 (NASDAQ) or 1,737 (NYSE), or dates that do not
match §2.2. In either case, stop and document before continuing.

### 2.4 Missing data

A stock-day enters the loss only if the close at `t` and the close at `t+1` are both observed.
The same mask applies to every model and placebo, so the comparison stays paired. Features that
need a missing lookback value are imputed with the cross-sectional median on that day. The
imputation is identical across models.

## 3. Prediction task

- **Decision time:** close of day `t`, using only prices up to `t`.
- **Target:** the next-day close-to-close return minus that day's equal-weighted cross-sectional
  mean (market excess; this matches the cross-sectionally demeaned DGP in the power study).
- **Primary loss:** daily cross-sectional MSE, with the target winsorised at the 0.5% and 99.5%
  quantiles of the **training** period (thresholds frozen before validation). Each model therefore
  produces one loss per test day, and every D quantity is a daily series.
- **Secondary:** unwinsorised MSE, MAE, Pearson IC, rank IC.

**Node features** (`d = 4`, computed at `t` and cross-sectionally z-scored per day):
1-day return, 5-day return, 20-day return, and 20-day realised volatility. The original RSR
moving-average features are not used. A stock-day is excluded from the training/evaluation mask
until it has a 20-day lookback.

## 4. Graphs and the placebo ladder

| Level | Graph | Retrained placebos (K) |
|---|---|---|
| L0 | none (own features only) | — |
| L1 | Erdős–Rényi with the same edge count | 3 |
| L2 | degree-preserving swaps (10 swaps per edge) | 3 |
| L3 | degree + same-industry-neighbour count preserved (intra-industry edges swapped within industry, inter-industry edges swapped among inter-industry edges) | **K_L3, see §6.3** |
| L4 | real relation graph | — |

The swap algorithm is exactly `dapilot/power.py::industry_edges` / `degree_edges`, already tested
in `tests/test_power.py`. Each placebo's diagnostics (edge overlap with L4, preservation check
L3 intra-count change = 0) are reported.

- **Industry labels for L3:** the RSR-supplied industry relation. It is not point-in-time, and it
  is labelled as such.
- **Real graph L4:**
  - **(a) Wiki relations — CONFIRMATORY.** All relation types are merged into one binary
    undirected graph, with self-loops excluded. This is the relation used in the RSR
    literature, so the test speaks directly to that literature. *Caveat:* the relations were
    collected after the sample period and are not point-in-time, which is possible look-ahead. A
    positive edge result has to be read with this in mind. A null result is not affected by it.
  - **(b) Return-dependence graph (G2 in the freeze) — SECONDARY/descriptive.** Mutual top-k
    absolute shrinkage correlation, estimated **only from the train period** (static), with
    k ∈ {5, 10, 20} chosen on validation using the ridge expert. Its timing is clean, but it is
    not the literature's graph.

  **Amendment A1 to `BENCHMARK_FREEZE.md` §4:** Wiki relations are added as the track-O
  confirmatory graph. They are not in the freeze's graph list.
- If the industry graph itself is used as L4, L3 becomes almost identical to the real graph, so
  the industry graph is only evaluated at the `industry` step (L2 → L3), descriptively.

## 5. Experts

**Primary expert (identical to the validated power study):** pooled ridge on
`[x, neighbour mean of x, neighbour sum of x / mean degree]` (`graph_features` in
`dapilot/power.py`). L0 uses `x` only. Features are standardised on train, α ∈ {1, 1e2, 1e4, 1e6}
is selected on validation **separately for every model and every placebo**, and the model is then
frozen and evaluated on test. There is no refitting on validation.

**Secondary expert (robustness, descriptive):** a matched two-layer GCN and a matched MLP (the GCN
without the neighbour pathway) as in `dapilot/experts.py`. They get the same budget and early
stopping on validation, 3 seeds, and K = 19 L3 placebos. GCN results do not enter the
confirmatory family, because their inference was not validated in the power study.

## 6. Inference

### 6.1 D1: circular block bootstrap, L = 20

Selected under the v2.1 binding commitment (size 0.028, Wilson upper 0.049, n = 400). Reject
H0: E[D1] ≤ 0 when the lower bound of the 90% two-sided bootstrap interval (1,000 repetitions)
is above 0. Equivalently, the one-sided p-value is the share of bootstrap means ≤ 0. Block lengths
40 and 60 are reported as sensitivity checks.

### 6.2 Edge: placebo-rank randomisation test

Selected in v2.1 (size 21/400 = 0.052, Wilson upper 0.079):

`p = (1 + #{k : mean test loss(L3_k) ≤ mean test loss(L4)}) / (K_L3 + 1)`

The block-bootstrap interval for `edge` is reported only as a descriptive interval, because its
size was 0.14–0.15.

### 6.3 Multiplicity and K_L3 = 99

With the validated K = 19, the smallest possible p-value is 1/20 = 0.05, so Holm across more than
one edge test could never reject. Therefore **K_L3 = 99**, which makes the smallest p 0.01.

**Deviation D-K from `configs/power_ladder_v2_1.yaml` (K = 19):** the placebo-rank test is exact for
any K under exchangeability, so its size does not depend on K. The v2.1 empirical size check
(21/400 at K = 19) supports the exchangeability assumption in this DGP. This change is motivated
only by the p-value resolution needed for Holm, and was decided before any RSR data were seen. The
99 placebos used for confirmation are generated with a fixed seed, all of them are reported, and
none are dropped.

Confirmatory family: 4 tests, `{D1, edge} × {NASDAQ, NYSE}`, with Holm at α = 0.05.

### 6.4 Markets

NASDAQ and NYSE are the confirmatory units and are reported separately, as the freeze requires.
The pooled panel (graphs are disjoint, so there are no cross-market edges) is secondary only.

## 7. Expected power (from the power study, optimistic)

Power is the rejection rate at R² = 1e-4 (implied daily IC ≈ 0.011), for a ridge expert that knows
the true functional form, before Holm correction, with a 252-day test period. Edge power is for K = 19; at K = 99 it is expected to be the same or slightly higher:

| Cell | D1 (bootstrap_L20) | edge (placebo_rank, K = 19) |
|---|---|---|
| NASDAQ | 0.69 | 0.98 |
| NYSE | 0.90 | 1.00 |
| Pooled | 0.98 | 1.00 |

The real Wiki graph is very different from the simulated graph (sparse, many isolated nodes, hubs,
and much larger industries), so these numbers are only a rough guide. The minimum detectable
effect (MDE80) is reported alongside every null result.

## 8. Confirmatory hypotheses

| ID | Market | H1 | Test |
|---|---|---|---|
| R1 | NASDAQ | E[D1] > 0 | bootstrap_L20, one-sided |
| R2 | NYSE | E[D1] > 0 | bootstrap_L20, one-sided |
| R3 | NASDAQ | edge > 0 | placebo_rank, K_L3 = 99 |
| R4 | NYSE | edge > 0 | placebo_rank, K_L3 = 99 |

Holm across R1–R4 at α = 0.05.

**Verdict labels:**
- **supported:** Holm-adjusted rejection.
- **not detected:** no rejection. Always reported with an effect size, an interval, and the MDE80.
  Never described as "no effect" or as equivalence.
- **opposite direction:** reported if the 90% bootstrap interval lies entirely below 0, as a
  descriptive finding.

**Interpretation rule, fixed in advance:** R3/R4 can be supported when R1/R2 are not. That outcome
means the real edges beat matched placebos even though the graph as a whole does not beat no
graph. It is reported as such, with no post-hoc reframing.

## 9. Secondary and exploratory analyses (not confirmatory)

1. Full ladder decomposition (`density`, `degree`, `industry`, `edge`) per market, with bootstrap
   intervals.
2. Pooled market.
3. Graph candidate (b), the train-period correlation graph.
4. GCN/MLP robustness (§5).
5. Secondary losses: unwinsorised MSE, MAE, IC, rank IC.
6. Conditional table: D1/edge by market-volatility and cross-sectional-dispersion terciles, with
   thresholds taken from train.
7. Bootstrap block-length sensitivity (L = 40, 60).

Everything in this list is labelled exploratory in every table.

## 10. Execution order

1. Freeze this document and the config, and hash them.
2. Download RSR, record provenance, run the intake checks (§2.3).
3. Build the pipeline and tests (leakage, maturity, mask pairing, L3 preservation) using **train
   and validation only**. Smoke test with the test period replaced by validation.
4. Run once on the test period. Generate the report automatically from the frozen config.
5. Record any deviation in a separate `DEVIATIONS.md`, with its timestamp and reason.

## 11. Known limitations (fixed in advance)

- Fixed universe with price/continuous-trading filters, so survivorship is likely. Results are
  conditional on the RSR universe.
- Wiki and industry relations are not point-in-time.
- One market history and one test year. Model seeds are not independent replications.
- The ridge expert is linear. A null result does not rule out nonlinear graph value.
- The test period has 237 days rather than the 252 used in the power study.

## Resolved decisions (v0.2, 2026-11-09, before any RSR data were seen)

1. **K_L3 = 99, Holm across 4 tests** (§6.3). This is deviation D-K.
2. **L4 = Wiki relations** (confirmatory). The train-period correlation graph is secondary (§4).
   This is amendment A1 to the freeze.
3. **Separate markets** (NASDAQ, NYSE) are the confirmatory units. Pooled is secondary (§6.4).
4. **Node features:** 1/5/20-day returns and 20-day realised volatility (§3).
