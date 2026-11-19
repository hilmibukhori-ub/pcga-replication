# P1 — power on the real RSR Wiki topology

See `configs/power_wiki_p1.yaml` and DEVIATIONS.md P1. No RSR returns used. Ridge knows the true functional form, so these are optimistic upper bounds.

`*_05` = rejection at 0.05; `*_holm` = rejection at 0.0125 (smallest Holm threshold, conservative).

| market | R² | IC | n | D1 @.05 | D1 @Holm | edge @.05 | edge @Holm |
|---|---|---|---|---|---|---|---|
| NASDAQ | 0 | 0.000 | 300 | 0.03 | 0.01 | 0.07 | 0.02 |
| NASDAQ | 0.0001 | 0.011 | 300 | 0.71 | 0.55 | 0.97 | 0.94 |
| NASDAQ | 0.0002 | 0.016 | 300 | 0.94 | 0.86 | 1.00 | 1.00 |
| NASDAQ | 0.0005 | 0.025 | 300 | 1.00 | 0.99 | 1.00 | 1.00 |
| NASDAQ | 0.001 | 0.036 | 300 | 1.00 | 1.00 | 1.00 | 1.00 |
| NASDAQ | 0.002 | 0.051 | 300 | 1.00 | 1.00 | 1.00 | 1.00 |
| NASDAQ | 0.005 | 0.080 | 300 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0 | 0.000 | 100 | 0.00 | 0.00 | 0.02 | 0.00 |
| NYSE | 0.0001 | 0.011 | 100 | 0.80 | 0.68 | 0.99 | 0.93 |
| NYSE | 0.0002 | 0.016 | 100 | 0.97 | 0.91 | 1.00 | 1.00 |
| NYSE | 0.0005 | 0.025 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0.001 | 0.035 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0.002 | 0.050 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |
| NYSE | 0.005 | 0.079 | 100 | 1.00 | 1.00 | 1.00 | 1.00 |

## Size at R² = 0 (Wilson 95% upper)

- NASDAQ D1: 9/300 = 0.030 (upper 0.056)
- NASDAQ edge: 20/300 = 0.067 (upper 0.101)
- NYSE D1: 0/100 = 0.000 (upper 0.037)
- NYSE edge: 2/100 = 0.020 (upper 0.070)

## MDE80 (smallest R² with power ≥ 0.80 at the Holm-worst level)

- NASDAQ D1: R² 0.0002 (IC ≈ 0.016)
- NASDAQ edge: R² 0.0001 (IC ≈ 0.011)
- NYSE D1: R² 0.0002 (IC ≈ 0.016)
- NYSE edge: R² 0.0001 (IC ≈ 0.011)

## L3 placebo edge overlap with Wiki (mean)

- NASDAQ: 0.55
- NYSE: 0.60

## P1b outcome (binding, per DEVIATIONS P1b) — 2026-11-10

NASDAQ edge size at n=300: 20/300 = 0.067, Wilson 95% upper 0.101 > 0.08.
→ R3 on NASDAQ is reported as **"calibration doubtful on this topology"**; a significant NASDAQ R3 is NOT claimed as evidence of edge value. NYSE R3, R1, R2, R4 unaffected. (Size at the Holm-worst level 0.0125 is 0.02 on NASDAQ.) P1 n=100 report archived as REPORT_p1_n100.md. Note: this block is lost if --summarise is rerun.
