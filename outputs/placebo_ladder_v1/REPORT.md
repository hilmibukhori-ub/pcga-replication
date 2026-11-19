# Placebo ladder v1 — synthetic screen

Pre-registered in `configs/placebo_ladder.yaml`. Ridge experts, 20 seeds, 10 retrained
placebos per level. Values are test-MSE differences x 1e3 (positive = structure helps).

## Ladder decomposition (mean ± s.e. over seeds)

| scenario | oracle_rel_gain | D1 | density | degree | block | edge | D2_er | D2_perm |
|---|---|---|---|---|---|---|---|---|
| edge | +20.262 ± 0.459 | +6.932 ± 0.264 | -0.007 ± 0.010 | +0.005 ± 0.016 | +0.562 ± 0.058 | +6.372 ± 0.234 | +6.939 ± 0.262 | +6.959 ± 0.263 |
| block | +14.049 ± 0.275 | +0.879 ± 0.060 | +0.008 ± 0.011 | -0.011 ± 0.016 | +0.882 ± 0.056 | -0.001 ± 0.028 | +0.871 ± 0.060 | +0.888 ± 0.055 |
| degree | +12.887 ± 0.483 | +9.888 ± 0.444 | +0.145 ± 0.024 | +9.823 ± 0.415 | +0.046 ± 0.057 | -0.126 ± 0.067 | +9.743 ± 0.431 | +9.762 ± 0.425 |
| null | +0.000 ± 0.000 | -0.022 ± 0.022 | -0.016 ± 0.009 | -0.013 ± 0.014 | +0.012 ± 0.011 | -0.005 ± 0.017 | -0.006 ± 0.026 | +0.016 ± 0.022 |

## Pre-registered hypotheses

| id | scenario | quantity | mean x1e3 / share | CI x1e3 | p_holm | verdict |
|---|---|---|---|---|---|---|
| H1 | edge | edge | +6.372 | [+5.882, +6.862] | 1.7e-05 | supported |
| H2 | block | block | +0.882 | [+0.765, +1.000] | 1.7e-05 | supported |
| H3 | degree | degree | +9.823 | [+8.955, +10.691] | 1.7e-05 | supported |
| H5a | block | D2_er | +0.871 | [+0.746, +0.995] | 1.7e-05 | supported |
| H5b | degree | D2_perm | +9.762 | [+8.871, +10.652] | 1.7e-05 | supported |
| H6_density | null | density | -0.016 | [-0.034, +0.002] | 0.28 | no clear difference |
| H6_degree | null | degree | -0.013 | [-0.043, +0.017] | 0.86 | no clear difference |
| H6_block | null | block | +0.012 | [-0.011, +0.036] | 0.74 | no clear difference |
| H6_edge | null | edge | -0.005 | [-0.041, +0.031] | 0.9 | no clear difference |
| H4_edge | edge | share(edge) | 0.91 (seeds ≥ thr: 100%) | | | supported |
| H4_block | block | share(block) | 0.91 (seeds ≥ thr: 95%) | | | supported |
| H4_degree | degree | share(degree) | 0.97 (seeds ≥ thr: 100%) | | | supported |

## Placebo sanity checks (mean over seeds)

- edges per graph: 152.2
- L2 (degree) total |change| in per-node block counts: 380.4
- L3 (degree+block) total |change| in per-node block counts: 0.0 (must be 0)
- L3 edge overlap with the real graph: 0.39
