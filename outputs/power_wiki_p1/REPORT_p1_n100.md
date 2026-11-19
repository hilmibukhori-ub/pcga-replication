# P1 — power on the real RSR Wiki topology

See `configs/power_wiki_p1.yaml` and DEVIATIONS.md P1. No RSR returns used. Ridge knows the true functional form, so these are optimistic upper bounds.

`*_05` = rejection at 0.05; `*_holm` = rejection at 0.0125 (smallest Holm threshold, conservative).

| market | R² | IC | n | D1 @.05 | D1 @Holm | edge @.05 | edge @Holm |
|---|---|---|---|---|---|---|---|
| NASDAQ | 0 | 0.000 | 100 | 0.03 | 0.01 | 0.09 | 0.03 |
| NASDAQ | 0.0001 | 0.012 | 100 | 0.76 | 0.62 | 0.98 | 0.97 |
| NASDAQ | 0.0002 | 0.016 | 100 | 0.96 | 0.89 | 0.99 | 0.99 |
| NASDAQ | 0.0005 | 0.026 | 100 | 1.00 | 0.99 | 1.00 | 1.00 |
| NASDAQ | 0.001 | 0.036 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NASDAQ | 0.002 | 0.051 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NASDAQ | 0.005 | 0.081 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0 | 0.000 | 100 | 0.00 | 0.00 | 0.02 | 0.00 |
| NYSE | 0.0001 | 0.011 | 100 | 0.80 | 0.68 | 0.99 | 0.93 |
| NYSE | 0.0002 | 0.016 | 100 | 0.97 | 0.91 | 1.00 | 1.00 |
| NYSE | 0.0005 | 0.025 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0.001 | 0.035 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0.002 | 0.050 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0.005 | 0.079 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |

## Size at R² = 0 (Wilson 95% upper)

- NASDAQ D1: 3/100 = 0.030 (upper 0.085)
- NASDAQ edge: 9/100 = 0.090 (upper 0.162)
- NYSE D1: 0/100 = 0.000 (upper 0.037)
- NYSE edge: 2/100 = 0.020 (upper 0.070)

## MDE80 (smallest R² with power ≥ 0.80 at the Holm-worst level)

- NASDAQ D1: R² 0.0002 (IC ≈ 0.016)
- NASDAQ edge: R² 0.0001 (IC ≈ 0.012)
- NYSE D1: R² 0.0002 (IC ≈ 0.016)
- NYSE edge: R² 0.0001 (IC ≈ 0.011)

## L3 placebo edge overlap with Wiki (mean)

- NASDAQ: 0.55
- NYSE: 0.60
