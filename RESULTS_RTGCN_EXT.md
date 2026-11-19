# RT-GCN extension (NYSE-U, NASDAQ-T, NYSE-T): confirmatory results

Pre-registration: `PREREG_RTGCN_EXT.md`, frozen 2026-11-10 18:34 before any main-confirm result was read.
Freeze record: `PREREG_RTGCN_EXT_FREEZE.md`. All 35 hashes were re-verified on hyperion on 2026-11-11.

Runs: DIPC hyperion, SLURM arrays 7944076–81. All 157 of 157 models finished (NYSE-U 69, NASDAQ-T 44,
NYSE-T 44), with 0 error logs.

Analysis: `dapilot.rtgcn_ext_report.summarise(configs/rtgcn_ext.yaml)` writes
`outputs/rtgcn_ext_v1/results.json` and `REPORT.md`. It was run once, on 2026-11-11 at about 08:40.

Setup: paper configuration, val arm. IRR-1 is the primary metric, computed over **all 172 test days**.

## Confirmatory family (Holm over 6 tests, α = 0.05)

| test | cell | estimate (IRR-1) | 95% CI | p | p (Holm) | seeds > 0 | verdict |
|---|---|---|---|---|---|---|---|
| H-A (test- vs val-selected epoch) | NYSE-U | +3.324 | [−1.471, +8.119] | 0.082 | 0.489 | 9/20 | not detected |
| H-B (D1: real vs none) | NYSE-U | −1.791 | [−5.416, +1.835] | 0.843 | 1.000 | 8/20 | not detected |
| H-A | NASDAQ-T | −0.093 | [−0.856, +0.670] | 0.606 | 1.000 | 4/10 | not detected |
| H-B | NASDAQ-T | −0.155 | [−0.488, +0.178] | 0.841 | 1.000 | 2/10 | not detected |
| H-A | NYSE-T | +3.307 | [−4.241, +10.854] | 0.174 | 0.707 | 4/10 | not detected |
| H-B | NYSE-T | +3.864 | [−3.787, +11.515] | 0.141 | 0.707 | 7/10 | not detected |

Mean IRR-1 at the validation-selected epoch:
- **NYSE-U:** real 2.25, none 4.04
- **NASDAQ-T:** real 0.27, none 0.43
- **NYSE-T:** real 4.88, none 1.02

**H-C (edge, exact rank, K = 19, seed 0, unadjusted):**

| cell | real | placebo mean | placebos ≥ real | p |
|---|---|---|---|---|
| NYSE-U | −0.006 | +0.212 | 15/19 | 0.80 |
| NASDAQ-T | +0.273 | +0.182 | 9/19 | 0.50 |
| NYSE-T | +2.967 | +2.810 | 7/19 | 0.40 |

## Interpretation (pre-registered rules, §8)

- **None of the six confirmatory tests is supported**, and no H-C test is significant.
- **Rule 2 applies.** The relation gain of the paper's best variant (T) does not replicate under a matched
  no-relation baseline, in either market.
  - The paper's gains over Rank_LSTM are +0.90 (NASDAQ) and +0.62 (NYSE) in IRR-1. Rank_LSTM is a
    different baseline from ours.
  - NASDAQ-T: our estimate is −0.16, with 95% CI [−0.49, +0.18], and only 2 of 10 seeds are positive.
  - NYSE-T: our estimate is +3.86, with CI [−3.79, +11.52]. Its size and width are driven by test day 89
    (see below).
- **NYSE-U:** D1 is negative (−1.79). For comparison, the paper's U gain over Rank_LSTM is +0.35
  (0.65 vs 0.30).
- **H-A is not detected in any cell.** On NASDAQ-T the inflation estimate is even negative.

## Test day 89 on NYSE (exploratory, post hoc; DEVIATIONS RX-S1)

This analysis was done after the confirmatory read and does not change the verdicts above.

- On NYSE, test day 89 contains one stock with a daily return of about **+3,321% (×33.2)**. This is the
  same day the released code deletes, with a comment about its influence on top-n metrics.
- A model that picks that stock on day 89 gains about 33 IRR-1. This happened in NYSE-U seed 6 (IRR-1
  33.8) and NYSE-T seed 3 (34.2).
- The NYSE between-seed sd of IRR-1 is 7.5 for NYSE-U and 10.3 for NYSE-T. On NASDAQ it is 0.69.

Results with day 89 excluded (`scripts/rtgcn_ext_sensitivity_day89.py` → `sensitivity_excl_day89.json`):

| cell | H-A | H-B (D1) | H-C | between-seed sd |
|---|---|---|---|---|
| NYSE-U | −0.002 [−0.34, +0.33], p 0.51 | −0.128 [−0.69, +0.43], p 0.68 (8/20) | 15/19 ≥ real, p 0.80 | 0.75 |
| NASDAQ-T | −0.119 [−0.87, +0.63], p 0.64 | −0.166 [−0.50, +0.17], p 0.85 (2/10) | 9/19, p 0.50 | 0.68 |
| NYSE-T | −0.023 [−0.97, +0.92], p 0.52 | +0.543 [−0.40, +1.49], p 0.11 (7/10) | 7/19, p 0.40 | 0.94 |

Without day 89, the NYSE variance falls to the NASDAQ level, and the qualitative conclusions are
unchanged. The largest D1 estimate (NYSE-T, +0.54) is not significant even before any correction, and
the real graph is no better than its placebos.

## Reproduction of published levels (R0: pub arm, test-selected, day 89 deleted)

| cell | IRR-1 (paper) | IRR-5 (paper) | IRR-10 (paper) | MRR (paper) |
|---|---|---|---|---|
| NYSE-U (10 seeds) | 0.61 [−0.17, 1.38] (0.65) | 0.67 [0.33, 1.00] (0.88) | 0.69 [0.57, 0.82] (0.96) **gap** | 0.041 [0.031, 0.051] (0.040) |
| NASDAQ-T (5 seeds) | −0.06 [−0.68, 0.57] (1.25) **gap** | 0.62 [0.21, 1.02] (0.97) | 0.67 [0.39, 0.95] (1.03) **gap** | 0.041 [0.033, 0.049] (0.061) **gap** |
| NYSE-T (5 seeds) | 0.58 [−0.84, 1.99] (0.92) | 0.47 [0.23, 0.71] (1.10) **gap** | 0.66 [0.52, 0.79] (1.13) **gap** | 0.038 [0.029, 0.047] (0.056) **gap** |

- **NYSE-U** largely reproduces. IRR-1 and MRR fall inside our intervals.
- **The time-aware variant does not reproduce.** The published T levels lie above our intervals on most
  metrics, in both markets.
- We tested the released code as it is (PREREG_RTGCN_EXT §1). That includes the float `attn_mask`,
  which adds a bias rather than masking (Appendix B of the paper draft). This is a plausible but
  untested reason for the gap. It is not claimed as the cause.

## Caveats

- There is one test year (2020). The T cells have 10 seeds for H-A/H-B and 5 for R0.
- The weight decay used (1e-4) is an assumption.
- NYSE-T and NASDAQ-T were run with 4 threads. Determinism holds for a fixed thread count.

> **Correction (2026-11-11, post hoc review):** the day-89 windfall (~33 IRR-1) also occurs in two
> no-relation runs of NYSE-U (none seeds 6 and 18), not only in the real-graph runs named above
> (NYSE-U real seed 6, NYSE-T real seed 3). It therefore pushes the NYSE-U D1 estimate down and the NYSE-T
> estimate up. See RESULTS_POSTHOC_REVIEW.md §2 for the data check (consistent with an unadjusted reverse
> split).
