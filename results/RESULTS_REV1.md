# Revision 1 robustness — results (PREREG_REV1.md)

Post hoc in motivation, pre-registered before any Revision-1 model was trained.

## Study S — RSR, industry x size-bin placebos (K = 99, Holm over 2 markets)

- **H-S NASDAQ**: real IRR -0.390, size-bin placebo mean -0.454 (sd 0.168); real ranks 30/100; p 0.300, Holm 0.420 → not detected
  - parent within-industry null, same T(real): rank 17/100, p 0.170
  - determinism: re-run real IRR -0.389818 vs parent -0.389818 (|diff| 0.00e+00, reproduced)
- **H-S NYSE**: real IRR +2.224, size-bin placebo mean +1.947 (sd 0.444); real ranks 21/100; p 0.210, Holm 0.420 → not detected
  - parent within-industry null, same T(real): rank 26/100, p 0.260
  - determinism: re-run real IRR +2.223619 vs parent +2.223619 (|diff| 0.00e+00, reproduced)

## Study K — RT-GCN(U), K = 99 (Holm over 2 cells)

- **H-C99 NASDAQ_U**: real IRR-1 +0.869, placebo mean +0.830 (sd 0.598); rank 31/100 (K=19: 9/20); p 0.310, Holm 0.620 → not detected
  - 80% power needs a shift of 2.56 placebo sd = 1.53 IRR-1 (α 0.05); 3.04 sd = 1.82 (Holm first step)
- **H-C99 NYSE_U**: real IRR-1 -0.006, placebo mean +0.532 (sd 3.552); rank 70/100 (K=19: 16/20); p 0.700, Holm 0.700 → not detected
  - 80% power needs a shift of 2.56 placebo sd = 9.10 IRR-1 (α 0.05); 3.04 sd = 10.78 (Holm first step)
  - descriptive, day 89 excluded: rank 70/100, p 0.700

## Notes added after the analysis (not part of the pre-registered output)

- **Platform (DEVIATIONS R1-1).** Every study-S model in the H-S tests was trained on hyperion. The determinism
  re-run of real s0 matches the parent laptop value exactly in both markets, so T(real) is the parent value.
- **Cross-platform check** (84 laptop/hyperion pairs, descriptive):
  - Val losses agree to ≤ 1.5e-8, and the selected epoch is the same in 84/84 pairs.
  - The top-1 IRR differs in 17/84 pairs (max |diff| 0.61), because round-off flips near-tied top-1 stock picks
    on some days.
  - This is a property of the top-1 metric, not a change of method.
- **Sanity checks.** The parent-null ranks (17/100 and 26/100) and the recomputed K = 19 RT-GCN ranks (9/20 → p 0.45;
  16/20 → p 0.80) reproduce the values reported in the paper.
