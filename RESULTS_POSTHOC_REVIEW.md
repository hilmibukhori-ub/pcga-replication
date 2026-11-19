# Post hoc analyses requested in internal review (2026-11-11)

These analyses are descriptive and post hoc. They change no confirmatory verdict.
Scripts: `scripts/power_posthoc.py` → `outputs/power_posthoc.json`; `scripts/rtgcn_day89_check.py` →
`outputs/rtgcn_ext_v1/day89_data_check.json`.

## 1. What could the tests detect? (sensitivity, not "observed power")

### D1 tests
Columns:
- **Upper95**: the one-sided 95% upper confidence bound on the gain.
- **MDE80**: the gain detectable with 80% power by a one-sided paired t-test, at α = 0.05 or at the Holm
  first-step level α/m, computed with the observed SD of the per-seed differences.

| Test | Estimate | Upper95 | MDE80 (α=.05) | MDE80 (Holm 1st) | Published gain (different baseline) |
|---|---|---|---|---|---|
| RSR NASDAQ | +0.119 | +0.241 | 0.183 | 0.233 | +0.51 |
| RSR NYSE | −0.070 | +0.081 | 0.226 | 0.288 | +0.23 |
| RT-GCN NASDAQ-U | −0.473 | +0.011 | 0.725 | 0.827 | +0.10 |
| RT-GCN NYSE-U | −1.791 | +1.204 | 4.486 | 6.038 | +0.35 |
| RT-GCN NASDAQ-T | −0.155 | +0.114 | 0.400 | 0.561 | +0.90 |
| RT-GCN NYSE-T | +3.864 | +10.064 | 9.187 | 12.908 | +0.62 |
| NYSE-U, day 89 excluded (exploratory) | −0.128 | +0.332 | 0.690 | 0.929 | +0.35 |
| NYSE-T, day 89 excluded (exploratory) | +0.543 | +1.309 | 1.135 | 1.595 | +0.62 |

Reading:
- **The RSR tests were sensitive at the scale of the published gains.** A matched gain of about 0.18–0.29
  IRR would be detected with 80% power.
- **The data are inconsistent with matched gains above these upper bounds:**
  - RSR NASDAQ: 0.24
  - RSR NYSE: 0.08
  - RT-GCN NASDAQ-U: 0.01
  - RT-GCN NASDAQ-T: 0.11
- **RT-GCN NASDAQ-U was underpowered for a gain as small as the published +0.10** (MDE 0.73), even
  though the upper bound of +0.01 lies below it.
- **The NYSE cells are uninformative about D1.** Their upper bounds (1.2 and 10.1, or 0.33 and 1.31
  without day 89) do not exclude the published gains.

### Edge rank tests
For 80% power, the real graph must sit this far above the placebo distribution. The calculation assumes
that real and placebo outcomes share the placebo spread (normal; simulation).

| Test | K | Shift needed (placebo SD) | Shift needed (IRR) |
|---|---|---|---|
| RSR NASDAQ | 99 | 2.54 | 0.54 |
| RSR NYSE | 99 | 2.54 | 1.00 |
| RT-GCN NASDAQ-U | 19 | 2.79 | 1.82 |
| RT-GCN NYSE-U | 19 | 2.79 | 0.88 |
| RT-GCN NASDAQ-T | 19 | 2.79 | 1.41 |
| RT-GCN NYSE-T | 19 | 2.79 | 1.20 |

Reading:
- The edge tests detect only large effects of the specific relations, about 2.5–2.8 placebo SDs.
- Smaller effects cannot be excluded.
- With K = 19, p ≤ 0.05 requires the real graph to beat all 19 placebos.
- Each graph is trained once, at seed 0.

## 2. The +3,321% NYSE stock-day (test day 89)

The released RT-GCN NYSE data contain no tickers or dates, so the stock cannot be identified by name.
Evidence from the released price arrays (stock index 1367 of 1,405):
- **The labels are next-day returns,** label[t] = price[t+1] / price[t] − 1, verified for all stocks and
  days in both markets.
- **The price had fallen to 0.54% of its 2015–2020 sample maximum.** In normalised prices it went from
  1.0 to about 0.0035. It then rose ×34.2 in one day and **stayed at the new level.** The 10-day median
  after the jump is 23.7 times the 10-day median before it, and the price never returns to its earlier
  level.
- **This pattern is typical of a reverse stock split that was not adjusted for in the price series,** not
  of a genuine one-day return. A collapse of more than 99% followed by a persistent jump of more than 30
  times fits a consolidation by a firm near the exchange's minimum price.
- **It is the only such event in the test data.**
  - The scan used day-on-day ratios above 5 or below 0.2, and kept those that persist (10-day level ratio
    above 4 or below 0.25).
  - NYSE has 1 event, NASDAQ has 0.
  - The next-largest test labels are +152% (NYSE) and +115% (NASDAQ).
- **The released code deletes this test day** (Appendix B). Its comment refers to the day's influence on
  top-n metrics, which suggests the authors had noticed the outlier.

What this means for the results:
- The confirmatory tests use all test days, as pre-registered.
- The exploratory re-analysis without day 89 (DEVIATIONS RX-S1) leaves every verdict unchanged.
- If the jump is a data error, the all-days NYSE estimates contain spurious gains of about 33 IRR-1 in
  every validation-selected run that ranked this stock first on day 89:
  - NYSE-U: real seed 6, and none seeds 6 and 18.
  - NYSE-T: real seed 3.
- These windfalls inflate the variance and make the NYSE all-days D1 estimates uninformative.
- Their sign is arbitrary, because it depends on which run happens to pick the stock. The windfalls push
  the NYSE-U D1 estimate down (net one extra windfall for none) and the NYSE-T estimate up (one windfall for
  real).
- Without day 89 the estimates are −0.13 (NYSE-U) and +0.54 (NYSE-T). Neither is detected.

Side note: there is also a train/test seam in the price arrays. Stock 218's price level differs by ×12
between the last training day and the first test day. Because the labels inside the test set are
consistent, this does not affect any test label.

## 3. Temporal robustness without retraining (sub-windows of the test period)

Script: `scripts/posthoc_subperiods_equivalence.py` → `outputs/posthoc_subperiods_equivalence.json`.
- Each test period is cut into four consecutive blocks: RSR 59–60 days each, RT-GCN 43 days each.
- D1 (paired over seeds) and the edge rank are recomputed per block from the stored daily top-1 returns of
  the epoch selected on validation.
- The same trained models are re-evaluated. Nothing is retrained.

Results:
- The full-period D1 is reproduced exactly in every study, which validates the script.
- **No block shows a nominally significant positive D1.** That is 0 of 24 blocks (6 studies × 4 blocks)
  with a one-sided p < 0.05 in favour of relations.
  - 13 of the 24 block estimates are positive.
  - **RSR NASDAQ is positive but small in all four blocks** (+0.01 to +0.05), which is consistent with its
    full-period p of 0.054.
- **Two blocks are nominally significant in favour of the network without relations,** and both are RT-GCN
  block 1, the start of the 2020 test period, which covers the COVID-19 crash:
  - NYSE-U: −0.43 [−0.80, −0.05]
  - NASDAQ-T: −0.35 [−0.61, −0.08]
- **The real graph never reaches the rejection region of its rank test in any block.**
  - RSR: the best rank is 10/100 (rejection needs ≤ 5).
  - RT-GCN: the best rank is 3/20 (rejection needs rank 1).
- The day-89 windfall falls in block 3 of the NYSE cells, so the large block-3 NYSE estimates are driven by
  it.

| Study | Block 1 | Block 2 | Block 3 | Block 4 |
|---|---|---|---|---|
| RSR NASDAQ (D1; edge rank /100) | +0.022; 23 | +0.010; 47 | +0.034; 10 | +0.053; 36 |
| RSR NYSE | +0.023; 12 | −0.053; 73 | −0.020; 47 | −0.019; 55 |
| RT-GCN NASDAQ-U (edge rank /20) | −0.227; 8 | −0.116; 20 | +0.092; 16 | −0.223; 8 |
| RT-GCN NYSE-U | −0.425*; 18 | +0.245; 16 | −1.688; 20 | +0.077; 8 |
| RT-GCN NASDAQ-T | −0.346*; 20 | −0.029; 18 | +0.010; 7 | +0.210; 8 |
| RT-GCN NYSE-T | −0.017; 12 | +0.207; 20 | +3.303; 20 | +0.370; 3 |

\* The 95% CI excludes 0 in favour of no relations. No correction is applied.

## 4. Equivalence and the published gain (post hoc)

- **TOST equivalence test** for |D1| < 0.09 IRR, which is the RSR paper's own "significant improvement"
  threshold. The bound was chosen after the read.
- **One-sided test of H0: D1 ≥ published gain.** The published gain is measured against a different
  baseline, so this test asks whether a *matched* gain of the published size is present.

| Study | 90% CI | TOST p | Equivalent? | p(H0: D1 ≥ published) |
|---|---|---|---|---|
| RSR NASDAQ | [−0.003, 0.241] | 0.66 | no | < 0.0001 (published 0.51) |
| RSR NYSE | [−0.221, 0.081] | 0.41 | no | 0.0014 (0.23) |
| RT-GCN NASDAQ-U | [−0.957, 0.011] | 0.91 | no | 0.027 (0.10) |
| RT-GCN NYSE-U | [−4.786, 1.204] | 0.83 | no | 0.12 (0.35) |
| RT-GCN NASDAQ-T | [−0.425, 0.114] | 0.67 | no | < 0.0001 (0.90) |
| RT-GCN NYSE-T | [−2.336, 10.064] | 0.85 | no | 0.82 (0.62) |

Reading:
- **No study establishes equivalence.** We cannot claim that the relations are practically useless.
- **Four of six reject a matched gain as large as the published one:** RSR in both markets, and RT-GCN
  NASDAQ-U and NASDAQ-T.
- **The NYSE RT-GCN cells reject neither,** because they are uninformative.

## 5. Audit: what is known about the 24 papers without full text

- 23 of the 24 have an abstract in the screening pool.
- No abstract mentions a random, shuffled, permuted or placebo graph, or a null model. Two mention only
  generic "ablation studies".
- Abstracts rarely describe controls, so this is weak evidence. It is the only evidence available without
  bypassing access controls.

Independent coding:
- A subset of 12 of the 36 full-text papers was drawn for blind second coding (`default_rng(20261111)`).
- The subset is in `literature/syssearch/rq2_double_coding_subset.csv`, and the protocol is in
  `literature/RQ2_DOUBLE_CODING_INSTRUCTIONS.md`.
- **Not yet coded:** this needs a second human coder.
