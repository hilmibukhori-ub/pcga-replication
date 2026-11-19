# Power analysis v2 — inference validity for the ladder `edge` step

Pre-registered in `configs/power_ladder_v2.yaml`. One-sided level 5% for every method.
Ridge expert knows the true functional form; powers are optimistic upper bounds.

Implied daily IC: R² 0 → 0.000, R² 5e-05 → 0.008, R² 0.0001 → 0.011, R² 0.0002 → 0.016, R² 0.0005 → 0.025

## Size at R² = 0 (all cells pooled) and pre-registered selection

| quantity | method | false positives | size | Wilson 95% upper | valid | mean RSR power at R² 1e-4 |
|---|---|---|---|---|---|---|
| D1 | bootstrap_L20 | 4/140 | 0.029 | 0.071 | yes | 0.85 |
| D1 | bootstrap_L40 | 5/140 | 0.036 | 0.081 | no | 0.86 |
| D1 | bootstrap_L60 | 5/140 | 0.036 | 0.081 | no | 0.85 |
| edge | bootstrap_L20 | 17/140 | 0.121 | 0.186 | no | 0.88 |
| edge | bootstrap_L40 | 17/140 | 0.121 | 0.186 | no | 0.87 |
| edge | bootstrap_L60 | 18/140 | 0.129 | 0.194 | no | 0.88 |
| edge | placebo_rank | 8/140 | 0.057 | 0.109 | no | 0.98 |

**`edge`: no method meets the validity rule.**
**Selected for `D1`: bootstrap_L20** (size 0.029, RSR power at 1e-4 0.85).

## Size by cell at R² = 0

| quantity | method | rsr_nasdaq | rsr_nyse | rsr_pooled | crsp_like |
|---|---|---|---|---|---|
| D1 | bootstrap_L20 | 0.03 (40) | 0.03 (40) | 0.03 (40) | 0.05 (20) |
| D1 | bootstrap_L40 | 0.03 (40) | 0.03 (40) | 0.05 (40) | 0.05 (20) |
| D1 | bootstrap_L60 | 0.03 (40) | 0.03 (40) | 0.05 (40) | 0.05 (20) |
| edge | bootstrap_L20 | 0.05 (40) | 0.07 (40) | 0.15 (40) | 0.30 (20) |
| edge | bootstrap_L40 | 0.05 (40) | 0.05 (40) | 0.17 (40) | 0.30 (20) |
| edge | bootstrap_L60 | 0.07 (40) | 0.07 (40) | 0.15 (40) | 0.30 (20) |
| edge | placebo_rank | 0.03 (40) | 0.05 (40) | 0.05 (40) | 0.15 (20) |

## Power: `edge`

| method | cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 |
|---|---|---|---|---|---|---|
| bootstrap_L20 | crsp_like | 0.30 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L20 | rsr_nasdaq | 0.05 | 0.35 | 0.72 | 0.95 | 1.00 |
| bootstrap_L20 | rsr_nyse | 0.07 | 0.68 | 0.90 | 1.00 | 1.00 |
| bootstrap_L20 | rsr_pooled | 0.15 | 0.93 | 1.00 | 1.00 | 1.00 |
| bootstrap_L40 | crsp_like | 0.30 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L40 | rsr_nasdaq | 0.05 | 0.38 | 0.72 | 0.93 | 1.00 |
| bootstrap_L40 | rsr_nyse | 0.05 | 0.68 | 0.88 | 1.00 | 1.00 |
| bootstrap_L40 | rsr_pooled | 0.17 | 0.88 | 1.00 | 1.00 | 1.00 |
| bootstrap_L60 | crsp_like | 0.30 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L60 | rsr_nasdaq | 0.07 | 0.42 | 0.78 | 0.93 | 1.00 |
| bootstrap_L60 | rsr_nyse | 0.07 | 0.75 | 0.88 | 0.97 | 1.00 |
| bootstrap_L60 | rsr_pooled | 0.15 | 0.93 | 1.00 | 1.00 | 1.00 |
| placebo_rank | crsp_like | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 |
| placebo_rank | rsr_nasdaq | 0.03 | 0.82 | 0.95 | 1.00 | 1.00 |
| placebo_rank | rsr_nyse | 0.05 | 0.93 | 1.00 | 1.00 | 1.00 |
| placebo_rank | rsr_pooled | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |

## Power: `D1`

| method | cell | 0 | 5e-05 | 0.0001 | 0.0002 | 0.0005 |
|---|---|---|---|---|---|---|
| bootstrap_L20 | crsp_like | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L20 | rsr_nasdaq | 0.03 | 0.35 | 0.65 | 0.93 | 1.00 |
| bootstrap_L20 | rsr_nyse | 0.03 | 0.68 | 0.90 | 1.00 | 1.00 |
| bootstrap_L20 | rsr_pooled | 0.03 | 0.93 | 1.00 | 1.00 | 1.00 |
| bootstrap_L40 | crsp_like | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L40 | rsr_nasdaq | 0.03 | 0.38 | 0.68 | 0.90 | 1.00 |
| bootstrap_L40 | rsr_nyse | 0.03 | 0.68 | 0.90 | 0.97 | 1.00 |
| bootstrap_L40 | rsr_pooled | 0.05 | 0.88 | 1.00 | 1.00 | 1.00 |
| bootstrap_L60 | crsp_like | 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |
| bootstrap_L60 | rsr_nasdaq | 0.03 | 0.40 | 0.68 | 0.90 | 1.00 |
| bootstrap_L60 | rsr_nyse | 0.03 | 0.72 | 0.88 | 0.97 | 1.00 |
| bootstrap_L60 | rsr_pooled | 0.05 | 0.90 | 1.00 | 1.00 | 1.00 |
