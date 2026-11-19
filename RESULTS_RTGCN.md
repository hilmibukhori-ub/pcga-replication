# RT-GCN re-test (Zheng et al., ICDE 2023): RT-GCN(U), NASDAQ. Confirmatory results

Pre-registration: `PREREG_RTGCN.md` (frozen 2026-11-10, with amendments RT1 and RT2 made before launch).
Runs: DIPC hyperion, SLURM array 7943650, 94/94 models, 0 errors. Analysis: `scripts/rtgcn_summarise.py` gives
`outputs/rtgcn_v1/REPORT.md` and `results.json`. Configuration: the paper config (lr 1e-3), val arm, 20 seeds,
IRR-1 over all 172 test days.

## Confirmatory family (Holm over 2, α = 0.05)

| test | estimate (IRR-1) | 95% CI | p | p (Holm) | seeds > 0 | verdict |
|---|---|---|---|---|---|---|
| **H-A** test-loss vs val-loss epoch selection (real) | **+0.284** (0.429 vs 0.144) | [−0.011, +0.580] | 0.029 | 0.059 | 10/20 | not detected |
| **H-B** D1: relations vs none (e_val) | **−0.473** (0.144 vs 0.617) | [−1.059, +0.113] | 0.946 | 0.946 | 7/20 | not detected |

**H-C edge** (exact placebo rank, K = 19, seed 0): real +0.869 against a placebo mean of +0.932. 8 of 19
placebos ≥ real, p = 0.45, so not detected.

## Interpretation (by the pre-registered rules in §8)

- **H-B.** The relation gain of RT-GCN(U) does not replicate under a matched no-relation baseline with
  validation-based epoch selection.
  - The point estimate is **negative**: the same network with A = I scores 0.62 ± 0.97, against 0.14 ± 0.79
    with relations.
  - The CI [−1.06, +0.11] also covers the paper's claimed +0.10 over Rank_LSTM. Rank_LSTM is a different
    baseline from ours, so this comparison is only indirect.
  - The secondary metrics point the same way: IRR-5 is 0.62 vs 0.79, IRR-10 is 0.79 vs 0.81, and MRR is
    0.048 vs 0.047.
- **H-A.** Selecting the epoch by test loss raised mean IRR-1 from 0.14 to 0.43, which is +0.28, or about
  3× the value under validation selection. This is significant without correction (p = 0.029) but not
  after Holm (0.059). Rule 2 applies: "loss-based selection does not *systematically* inflate IRR". The size
  of the effect is reported here and is comparable to the paper's claimed relation gain (+0.10). Within-run
  sd across epochs is 0.70, and between-seed sd is 0.79. The oracle best epoch averages **2.32**, so the
  scope for selection effects is large.
- **H-C.** The real Wiki assignment is no better than within-industry placebos.
- **R0 (published protocol, 10 seeds, day 89 deleted).** IRR-1 0.35 [−0.48, 1.19] (paper 0.45), IRR-5 0.81
  [0.59, 1.04] (paper 0.71), IRR-10 0.90 [0.75, 1.05] (paper 0.82), MRR 0.050 [0.038, 0.061] (paper 0.055).
  **There is no reproduction gap:** the paper's RT-GCN(U) numbers are reproduced within our CIs. Single-run
  sd of IRR-1 is 1.17, so one run can land almost anywhere.

**Overall:** the published RT-GCN(U) level reproduces, but the attribution to relations does not. A
matched no-relation network does at least as well. This matches the RSR re-test (`RESULTS_RSR.md`), where
H1 and H2 were not detected.

## Descriptive and sensitivity results (not confirmatory)

- **Released config (lr 0.1, wd 0.1), 10 seeds.**
  - D1 is **+0.290** [+0.011, +0.570], p 0.022, with 7/10 positive. Under the unstable released config the
    relations *do* show a positive descriptive difference.
  - Inflation from selection is +0.136 [−0.17, +0.44].
  - This is reported as found. It is descriptive, with no multiplicity control. It runs opposite to the
    primary config, which is evidence that the sign of D1 depends on the training configuration.
- **Industry-only D1** (5 seeds): −0.23 [−3.16, +2.71], uninformative.
- **Selected epochs.** Validation selection picks late epochs (76–99). Test-loss selection ranges from 1 to
  98.
- Last-epoch IRR-1: real −0.01 ± 0.53, none 0.15 ± 0.72.

## Caveats

- One market (NASDAQ), one test year (2020, including the COVID crash), variant U only. The pre-registered
  extension (NYSE-U, NASDAQ-T, NYSE-T; `PREREG_RTGCN_EXT.md`) is running.
- The paper's weight decay is unknown (assumed 1e-4). The released test set has 172 days against the
  paper's 207.
- The seed-paired t was A/A-calibrated in the RSR rehearsal, not for RT-GCN.
