# Power analysis v2 — inference validity for the ladder `edge` step

Pre-registered in `configs/power_ladder_v2.yaml`. One-sided level 5% for every method.
Ridge expert knows the true functional form; powers are optimistic upper bounds.

Implied daily IC: R² 0 → 0.000, R² 5e-05 → 0.008, R² 0.0001 → 0.011, R² 0.0002 → 0.016, R² 0.0005 → 0.025

## Size at R² = 0 (all cells pooled) and pre-registered selection

| quantity | method | false positives | size | Wilson 95% upper | valid | mean RSR power at R² 1e-4 |
|---|---|---|---|---|---|---|
| D1 | bootstrap_L20 | 11/400 | 0.028 | 0.049 | yes | 0.86 |
| D1 | bootstrap_L40 | 13/400 | 0.033 | 0.055 | yes | 0.86 |
| D1 | bootstrap_L60 | 14/400 | 0.035 | 0.058 | yes | 0.86 |
| edge | bootstrap_L20 | 55/400 | 0.138 | 0.175 | no | 0.89 |
| edge | bootstrap_L40 | 57/400 | 0.142 | 0.180 | no | 0.88 |
| edge | bootstrap_L60 | 59/400 | 0.147 | 0.186 | no | 0.89 |
| edge | placebo_rank | 21/400 | 0.052 | 0.079 | yes | 0.99 |

**Selected for `edge`: placebo_rank** (size 0.052, RSR power at 1e-4 0.99).
**Selected for `D1`: bootstrap_L40** (size 0.033, RSR power at 1e-4 0.86).

## Size by cell at R² = 0

| quantity | method | rsr_nasdaq | rsr_nyse | rsr_pooled | crsp_like |
|---|---|---|---|---|---|
| D1 | bootstrap_L20 | 0.02 (100) | 0.06 (100) | 0.02 (100) | 0.01 (100) |
| D1 | bootstrap_L40 | 0.02 (100) | 0.06 (100) | 0.04 (100) | 0.01 (100) |
| D1 | bootstrap_L60 | 0.02 (100) | 0.07 (100) | 0.04 (100) | 0.01 (100) |
| edge | bootstrap_L20 | 0.14 (100) | 0.11 (100) | 0.13 (100) | 0.17 (100) |
| edge | bootstrap_L40 | 0.15 (100) | 0.11 (100) | 0.13 (100) | 0.18 (100) |
| edge | bootstrap_L60 | 0.17 (100) | 0.12 (100) | 0.13 (100) | 0.17 (100) |
| edge | placebo_rank | 0.07 (100) | 0.04 (100) | 0.05 (100) | 0.05 (100) |

## Power: `edge`

| method | cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 |
|---|---|---|---|---|---|---|
| bootstrap_L20 | crsp_like | 0.17 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L20 | rsr_nasdaq | 0.14 | 0.44 | 0.79 | 0.97 | 1.00 |
| bootstrap_L20 | rsr_nyse | 0.11 | 0.71 | 0.91 | 0.97 | 1.00 |
| bootstrap_L20 | rsr_pooled | 0.13 | 0.85 | 0.98 | 1.00 | 1.00 |
| bootstrap_L40 | crsp_like | 0.18 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L40 | rsr_nasdaq | 0.15 | 0.48 | 0.76 | 0.97 | 1.00 |
| bootstrap_L40 | rsr_nyse | 0.11 | 0.74 | 0.90 | 0.97 | 1.00 |
| bootstrap_L40 | rsr_pooled | 0.13 | 0.85 | 0.98 | 1.00 | 1.00 |
| bootstrap_L60 | crsp_like | 0.17 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L60 | rsr_nasdaq | 0.17 | 0.52 | 0.79 | 0.97 | 1.00 |
| bootstrap_L60 | rsr_nyse | 0.12 | 0.79 | 0.89 | 0.96 | 1.00 |
| bootstrap_L60 | rsr_pooled | 0.13 | 0.87 | 0.98 | 1.00 | 1.00 |
| placebo_rank | crsp_like | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |
| placebo_rank | rsr_nasdaq | 0.07 | 0.85 | 0.98 | 1.00 | 1.00 |
| placebo_rank | rsr_nyse | 0.04 | 0.92 | 1.00 | 1.00 | 1.00 |
| placebo_rank | rsr_pooled | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |

## Power: `D1`

| method | cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 |
|---|---|---|---|---|---|---|
| bootstrap_L20 | crsp_like | 0.01 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L20 | rsr_nasdaq | 0.02 | 0.43 | 0.69 | 0.95 | 1.00 |
| bootstrap_L20 | rsr_nyse | 0.06 | 0.68 | 0.90 | 0.97 | 1.00 |
| bootstrap_L20 | rsr_pooled | 0.02 | 0.82 | 0.98 | 1.00 | 1.00 |
| bootstrap_L40 | crsp_like | 0.01 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L40 | rsr_nasdaq | 0.02 | 0.44 | 0.71 | 0.95 | 1.00 |
| bootstrap_L40 | rsr_nyse | 0.06 | 0.70 | 0.90 | 0.96 | 1.00 |
| bootstrap_L40 | rsr_pooled | 0.04 | 0.81 | 0.98 | 1.00 | 1.00 |
| bootstrap_L60 | crsp_like | 0.01 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L60 | rsr_nasdaq | 0.02 | 0.43 | 0.71 | 0.96 | 1.00 |
| bootstrap_L60 | rsr_nyse | 0.07 | 0.74 | 0.89 | 0.96 | 1.00 |
| bootstrap_L60 | rsr_pooled | 0.04 | 0.87 | 0.98 | 1.00 | 1.00 |

## Binding decision (v2.1, n = 400) — added 2026-11-09

- `edge`: **placebo_rank is valid** (size 21/400 = 0.052, Wilson upper 0.079 ≤ 0.08). It becomes the primary confirmatory inference for edge/D2 in the RSR pre-registration. The margin to the 0.08 bound is thin (0.001), and that should be reported. Per-cell size is 0.04–0.07, with crsp_like at 0.05. This is a secondary check and does not change the selection.
- `D1`: **bootstrap_L20 is kept**, as commitment 2 in `configs/power_ladder_v2_1.yaml` requires, because it is still valid (size 0.028, upper 0.049). The automatic line above picks bootstrap_L40 only because its mean power is marginally higher. Under the commitment, that automatic pick does not apply.
- Block bootstrap for `edge` (size 0.14–0.15) is reported only as a descriptive interval.
