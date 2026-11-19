# Power analysis — placebo ladder at RSR / CRSP sizes

Pre-registered in `configs/power_ladder.yaml`. Power = share of replicates whose 95%
circular block-bootstrap CI (20-day blocks) lies above 0. Ridge expert knows the true
functional form and no Holm correction is applied, so these are OPTIMISTIC upper bounds.

Implied daily cross-sectional IC of the graph signal: R² 0 → IC 0.000, R² 5e-05 → IC 0.008, R² 0.0001 → IC 0.011, R² 0.0002 → IC 0.015, R² 0.0005 → IC 0.024, R² 0.001 → IC 0.034, R² 0.002 → IC 0.049

## Power: `D1`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| rsr_nasdaq | 0.03 | 0.38 | 0.57 | 0.85 | 0.97 | 1.00 | 1.00 |

## Power: `edge`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| rsr_nasdaq | 0.00 | 0.42 | 0.60 | 0.90 | 0.97 | 1.00 | 1.00 |

## Power: `industry`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| rsr_nasdaq | 0.12 | 0.12 | 0.12 | 0.17 | 0.25 | 0.45 | 0.75 |

## Power: `degree`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| rsr_nasdaq | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

## Power: `density`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| rsr_nasdaq | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## MDE80 (smallest tested R² with power ≥ 0.80; NaN = none within grid)

| cell       |     D1 |   degree |   density |   edge |   industry |
|:-----------|-------:|---------:|----------:|-------:|-----------:|
| rsr_nasdaq | 0.0002 |      nan |       nan | 0.0002 |        nan |

## Pre-registered decision

Best RSR MDE80 for `edge`: 0.0002. Rule: proceed to RSR edge-level tests only if ≤ 5e-4.
**Decision: RSR can test edge-level claims.**

## Placebo checks

- max change in same-industry neighbour counts under L3: 0 (must be 0)
- L3 edge overlap with the real graph (mean by cell): rsr_nasdaq 0.25
