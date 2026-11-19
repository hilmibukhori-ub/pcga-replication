# Power analysis — placebo ladder at RSR / CRSP sizes

Pre-registered in `configs/power_ladder.yaml`. Power = share of replicates whose 95%
circular block-bootstrap CI (20-day blocks) lies above 0. Ridge expert knows the true
functional form and no Holm correction is applied, so these are OPTIMISTIC upper bounds.

Implied daily cross-sectional IC of the graph signal: R² 0 → IC 0.000, R² 5e-05 → IC 0.008, R² 0.0001 → IC 0.011, R² 0.0002 → IC 0.016, R² 0.0005 → IC 0.025, R² 0.001 → IC 0.036, R² 0.002 → IC 0.051

## Power: `D1`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| crsp_like | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Power: `edge`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| crsp_like | 0.12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Power: `industry`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| crsp_like | 0.05 | 0.28 | 0.55 | 0.90 | 1.00 | 1.00 | 1.00 |

## Power: `degree`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| crsp_like | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |

## Power: `density`

| cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|---|---|---|
| crsp_like | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## MDE80 (smallest tested R² with power ≥ 0.80; NaN = none within grid)

| cell      |    D1 |   degree |   density |   edge |   industry |
|:----------|------:|---------:|----------:|-------:|-----------:|
| crsp_like | 5e-05 |      nan |       nan |  5e-05 |     0.0002 |

## Placebo checks

- max change in same-industry neighbour counts under L3: 0 (must be 0)
- L3 edge overlap with the real graph (mean by cell): crsp_like 0.25
