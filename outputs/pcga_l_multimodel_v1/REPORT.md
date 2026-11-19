# PCGA-L cross-model construct recovery

Prospective protocol: `PREREG_PCGA_L_MULTIMODEL.md`. Confirmatory DGP seeds 100--119; 
all graph variants retrained; positive values mean that the added structure lowers test loss.

**Global pre-specified recovery criterion: NOT MET.**

## Signal-scenario recovery

| family | scenario | expected | mean step | expected share | seeds >= 70% | p Holm | directional test | concentration |
|---|---|---|---:|---:|---:|---:|---|---|
| ridge | edge | edge | +6.367e-03 | 0.909 | 100% | 4.01e-05 | supported | met |
| ridge | block | block | +8.746e-04 | 0.806 | 75% | 4.01e-05 | supported | not met |
| ridge | degree | degree | +9.685e-03 | 0.957 | 100% | 4.01e-05 | supported | met |
| mlp | edge | edge | +6.328e-03 | 0.906 | 100% | 4.01e-05 | supported | met |
| mlp | block | block | +8.046e-04 | 0.725 | 65% | 4.01e-05 | supported | not met |
| mlp | degree | degree | +9.613e-03 | 0.961 | 100% | 4.01e-05 | supported | met |
| gcn | edge | edge | +1.404e-02 | 0.949 | 100% | 4.01e-05 | supported | met |
| gcn | block | block | +8.475e-04 | 0.790 | 65% | 4.01e-05 | supported | not met |
| gcn | degree | degree | +1.109e-02 | 0.972 | 100% | 4.01e-05 | supported | met |

## Null negative controls

| family | step | mean | p Holm | verdict |
|---|---|---:|---:|---|
| ridge | density | -2.394e-05 | 1 | no clear difference |
| ridge | degree | +1.285e-05 | 1 | no clear difference |
| ridge | block | -1.833e-05 | 1 | no clear difference |
| ridge | edge | +2.904e-05 | 0.683 | no clear difference |
| mlp | density | -5.224e-04 | 4.01e-05 | difference detected |
| mlp | degree | +8.145e-05 | 0.532 | no clear difference |
| mlp | block | -5.769e-06 | 1 | no clear difference |
| mlp | edge | +5.788e-05 | 1 | no clear difference |
| gcn | density | -7.835e-04 | 4.01e-05 | difference detected |
| gcn | degree | +4.683e-05 | 1 | no clear difference |
| gcn | block | +2.181e-05 | 1 | no clear difference |
| gcn | edge | +5.115e-05 | 1 | no clear difference |

Null non-rejection is not equivalence. This experiment establishes construct recovery only 
for the fixed synthetic worlds and matched experts; it is not market or causal evidence.
