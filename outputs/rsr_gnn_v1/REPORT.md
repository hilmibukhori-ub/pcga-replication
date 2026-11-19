# RSR GCN/MLP robustness (EXPLORATORY, descriptive only)

PREREG_RSR.md §5/§9.4; spec in configs/rsr_gnn.yaml; DEVIATIONS G1. Not confirmatory. Positive = graph model has lower loss. Losses averaged over seeds before differencing. Edge rank p uses K = 19 (min p = 0.05); intervals: circular block bootstrap L=20, 90%.

## NASDAQ

Graphs complete: 27 (L3: 19, L1: 3, L2: 3); test 2017-01-03 → 2017-12-07

| quantity | estimate | rel. to L4 MSE | 90% interval | p | test |
|---|---|---|---|---|---|
| D1 (MLP − GCN-Wiki) | +2.11e-08 | +0.0090% | [-1.52e-08, +5.80e-08] | 0.158 | bootstrap |
| edge (L3 mean − Wiki) | +2.25e-08 | +0.0096% | [+4.17e-09, +4.02e-08] | 0.050 | rank, 0/19 placebos better |

Ladder (point estimates): density -9.56e-09, degree +6.45e-09, industry +1.62e-09, edge +2.25e-08

Per seed (D1 / edge, MSE): s0: +2.74e-08 / +2.31e-08; s1: +1.61e-08 / +2.57e-08; s2: +1.96e-08 / +1.87e-08

IC (seed-avg prediction losses, mean daily Pearson): MLP 0.0153, GCN-Wiki 0.0153, GCN-L3 mean 0.0148

Secondary losses D1: mse_raw +1.82e-08, mae +1.44e-06; edge: mse_raw +2.96e-08, mae +7.32e-07

Best epoch: median 22, min 4, max 69 (max_epochs 150)

## NYSE

Graphs complete: 27 (L3: 19, L1: 3, L2: 3); test 2017-01-03 → 2017-12-07

| quantity | estimate | rel. to L4 MSE | 90% interval | p | test |
|---|---|---|---|---|---|
| D1 (MLP − GCN-Wiki) | -2.72e-08 | -0.0184% | [-4.78e-08, -6.07e-09] | 0.984 | bootstrap |
| edge (L3 mean − Wiki) | -1.44e-08 | -0.0098% | [-2.57e-08, -3.00e-09] | 0.900 | rank, 17/19 placebos better |

Ladder (point estimates): density -2.53e-08, degree +8.82e-09, industry +3.65e-09, edge -1.44e-08

Per seed (D1 / edge, MSE): s0: -6.57e-08 / -3.83e-08; s1: -4.18e-08 / -2.20e-08; s2: +2.59e-08 / +1.72e-08

IC (seed-avg prediction losses, mean daily Pearson): MLP 0.0172, GCN-Wiki 0.0136, GCN-L3 mean 0.0153

Secondary losses D1: mse_raw -3.44e-08, mae -1.35e-06; edge: mse_raw -1.67e-08, mae -7.87e-07

Best epoch: median 14, min 0, max 94 (max_epochs 150)
