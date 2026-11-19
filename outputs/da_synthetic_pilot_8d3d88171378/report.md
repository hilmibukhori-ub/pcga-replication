# D + A synthetic pilot — results report

Run: `da_synthetic_pilot_8d3d88171378` · config hash `8d3d88171378` · seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

All numbers are locked-test (time steps 2001–3000) and aggregated over seeds; intervals are 95% t-intervals over seeds (n = 10). Hypothesis verdicts use a two-sided Wilcoxon signed-rank test on per-seed paired differences, Holm-adjusted within each family (D, A), plus the requirement that the t-interval excludes zero in the stated direction.

## Hypotheses

| ID | Hypothesis | Mean diff | 95% CI | Wilcoxon p (Holm) | +/− seeds | Verdict |
|---|---|---|---|---|---|---|
| HD1 | Graph value D1 (GCN vs matched MLP) > 0 in informative regime | 0.2663 | [0.2296, 0.3029] | 0.0117 | 10/0 | **supported** |
| HD2 | D1 < 0 in no-signal regime (graph hurts without relational signal) | -0.1279 | [-0.1531, -0.1026] | 0.0117 | 0/10 | **supported** |
| HD3 | D1 lower under corrupted adjacency than informative | -0.2417 | [-0.2939, -0.1896] | 0.0117 | 0/10 | **supported** |
| HD4 | D2 (placebo minus real graph) > 0 in informative regime | 0.2516 | [0.2281, 0.2752] | 0.0117 | 10/0 | **supported** |
| HD5 | D2 lower under corrupted adjacency than informative | -0.2336 | [-0.2730, -0.1943] | 0.0117 | 0/10 | **supported** |
| HD6 | GCN beats cross-asset non-graph control in informative regime | 0.1718 | [0.1475, 0.1961] | 0.0117 | 10/0 | **supported** |
| HA1 | A2a regret < A1 regret | -0.0077 | [-0.0429, 0.0275] | 1.0000 | 7/3 | **inconclusive** |
| HA2 | A2a regret < A2a without structural diagnostics | 0.0050 | [-0.0026, 0.0126] | 0.4219 | 7/3 | **inconclusive** |
| HA3 | A2a regret < A2a without D2 auxiliary task | -0.0033 | [-0.0108, 0.0041] | 1.0000 | 5/5 | **inconclusive** |
| HA4 | A2a regret < generic gate (same features) | 0.0186 | [0.0053, 0.0320] | 0.0488 | 9/1 | **contradicted** |
| HA5 | A2a regret < conditional-risk selector (same features) | -0.0150 | [-0.0330, 0.0031] | 0.4219 | 3/7 | **inconclusive** |
| HA6 | A2a with real graph attains lower conditional risk than with placebo graph | 0.0601 | [0.0319, 0.0883] | 0.0117 | 10/0 | **supported** |
| HA7 | A2b (dual evidence) vs A2a regret [exploratory, two-sided] | 0.0185 | [-0.0081, 0.0450] | 0.1602 | 8/2 | **no clear difference** |

Sign convention: D rows are the quantity itself (or a regime difference); A rows are regret(A) − regret(B), so negative favours the first-named method. HA6 is cond. risk with placebo-graph slot minus real-graph slot.

## Selector regret by regime (real graph slot)

| Method | all | informative | no_signal | corrupted | graph weight (all) | oracle agreement |
|---|---|---|---|---|---|---|
| AlwaysN | 0.1058 ± 0.0270 | 0.2676 ± 0.0407 | 0.0004 ± 0.0006 | 0.0751 ± 0.0239 | 0.00 | 0.47 |
| AlwaysG | 0.0503 ± 0.0096 | 0.0004 ± 0.0005 | 0.1164 ± 0.0133 | 0.0327 ± 0.0153 | 1.00 | 0.53 |
| FixedEnsemble | 0.0430 ± 0.0094 | 0.0461 ± 0.0315 | 0.0628 ± 0.0183 | 0.0189 ± 0.0095 | 0.75 | – |
| Gate | 0.0238 ± 0.0112 | 0.0372 ± 0.0226 | 0.0218 ± 0.0098 | 0.0133 ± 0.0031 | 0.57 | – |
| CondRisk | 0.0574 ± 0.0214 | 0.0768 ± 0.0632 | 0.0617 ± 0.0211 | 0.0443 ± 0.0133 | 0.63 | 0.58 |
| A1 | 0.0502 ± 0.0291 | 0.0671 ± 0.0915 | 0.0518 ± 0.0296 | 0.0399 ± 0.0210 | 0.61 | 0.63 |
| A2a | 0.0425 ± 0.0105 | 0.0275 ± 0.0360 | 0.0668 ± 0.0324 | 0.0385 ± 0.0152 | 0.74 | 0.63 |
| A2a_noStruct | 0.0375 ± 0.0114 | 0.0141 ± 0.0151 | 0.0609 ± 0.0266 | 0.0329 ± 0.0115 | 0.74 | 0.67 |
| A2a_noAux | 0.0458 ± 0.0144 | 0.0505 ± 0.0749 | 0.0645 ± 0.0318 | 0.0372 ± 0.0166 | 0.70 | 0.63 |
| A2b | 0.0609 ± 0.0268 | 0.1052 ± 0.0934 | 0.0443 ± 0.0319 | 0.0463 ± 0.0179 | 0.52 | 0.59 |
| Oracle | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.53 | 1.00 |

## Graph value (D1, D2) by regime

| Quantity | all | informative | no_signal | corrupted |
|---|---|---|---|---|
| D1_realised_vs_mlp | 0.0471 [0.012, 0.082] | 0.2663 [0.230, 0.303] | -0.1279 [-0.153, -0.103] | 0.0245 [-0.016, 0.065] |
| D1_oracle_vs_mlp | 0.0555 [0.022, 0.089] | 0.2672 [0.226, 0.308] | -0.1160 [-0.129, -0.103] | 0.0424 [0.006, 0.079] |
| D1_realised_vs_cross_mlp | 0.0282 [0.000, 0.056] | 0.1718 [0.148, 0.196] | 0.0019 [-0.032, 0.036] | -0.0534 [-0.102, -0.005] |
| D1_realised_vs_ridge | 0.1665 [0.114, 0.219] | 0.3991 [0.354, 0.444] | -0.0249 [-0.081, 0.032] | 0.1515 [0.098, 0.205] |
| D1placebo_realised_vs_mlp | -0.0053 [-0.016, 0.006] | 0.0146 [-0.005, 0.034] | -0.0391 [-0.046, -0.032] | 0.0065 [-0.006, 0.019] |
| D2_realised | 0.0523 [0.018, 0.087] | 0.2516 [0.228, 0.275] | -0.0888 [-0.117, -0.061] | 0.0180 [-0.021, 0.057] |
| D2_oracle | 0.0598 [0.030, 0.089] | 0.2560 [0.231, 0.281] | -0.0782 [-0.099, -0.058] | 0.0304 [0.000, 0.061] |

## Graph-value estimation error and calibration (real slot, all test steps)

| Method | benefit RMSE vs B* | corr with B* | calib. slope (median) | calib. slope (mean) | calib. intercept (median) | binned calib. error | D2 RMSE |
|---|---|---|---|---|---|---|---|
| CondRisk | 0.303 | 0.168 | 0.109 | 0.276 | 0.022 | 0.183 | – |
| A1 | 0.186 | 0.562 | 1.253 | 10.563 | -0.041 | 0.074 | – |
| A2a | 0.187 | 0.498 | 0.979 | 1.331 | -0.028 | 0.074 | 0.140 |
| A2a_noStruct | 0.179 | 0.523 | 0.876 | 0.909 | -0.010 | 0.065 | 0.136 |
| A2a_noAux | 0.190 | 0.490 | 0.914 | 22.894 | -0.001 | 0.077 | – |

Calibration slope = OLS slope of true benefit B* on the calibrated prediction (ideal 1). The seed mean is unstable: in seeds where a selector's test-period predictions are nearly constant the slope explodes (per-seed values in `per_seed_results.csv`), so the median is the robust summary.

## Ablation summary (regret A − B, real slot)

| Comparison | all | informative | no_signal | corrupted |
|---|---|---|---|---|
| A1 vs A2a | -0.0077 [-0.0429, 0.0275] | -0.0397 [-0.1108, 0.0314] | 0.0150 [-0.0054, 0.0354] | -0.0014 [-0.0185, 0.0156] |
| A2a: with vs without structural diagnostics | 0.0050 [-0.0026, 0.0126] | 0.0134 [-0.0209, 0.0476] | 0.0059 [-0.0051, 0.0169] | 0.0055 [-0.0076, 0.0186] |
| A2a: with vs without D2 auxiliary | -0.0033 [-0.0108, 0.0041] | -0.0231 [-0.0622, 0.0161] | 0.0023 [-0.0038, 0.0083] | 0.0013 [-0.0043, 0.0069] |
| A2a vs generic gate | 0.0186 [0.0053, 0.0320] | -0.0097 [-0.0481, 0.0286] | 0.0450 [0.0120, 0.0780] | 0.0252 [0.0116, 0.0387] |
| A2a vs conditional-risk | -0.0150 [-0.0330, 0.0031] | -0.0493 [-0.0861, -0.0126] | 0.0051 [-0.0194, 0.0295] | -0.0058 [-0.0152, 0.0036] |
| A2b vs A2a | 0.0185 [-0.0081, 0.0450] | 0.0778 [0.0112, 0.1444] | -0.0225 [-0.0447, -0.0004] | 0.0078 [-0.0056, 0.0213] |
| A1 vs A2a_noStruct (model class only) | -0.0127 [-0.0488, 0.0235] | -0.0531 [-0.1463, 0.0401] | 0.0091 [-0.0051, 0.0233] | -0.0070 [-0.0308, 0.0168] |
| A2a vs AlwaysN | -0.0633 [-0.0920, -0.0347] | -0.2401 [-0.2783, -0.2020] | 0.0664 [0.0339, 0.0988] | -0.0366 [-0.0659, -0.0074] |
| A2a vs AlwaysG | -0.0078 [-0.0162, 0.0005] | 0.0271 [-0.0090, 0.0632] | -0.0496 [-0.0776, -0.0216] | 0.0058 [-0.0111, 0.0227] |
| A2a vs FixedEnsemble | -0.0005 [-0.0109, 0.0099] | -0.0186 [-0.0676, 0.0304] | 0.0040 [-0.0341, 0.0421] | 0.0196 [0.0051, 0.0341] |

## Plots

- `plots/regret_by_regime.png`
- `plots/regret_by_regime_placebo.png`
- `plots/graph_value_by_regime.png`
- `plots/selection_frequency.png`
- `plots/hypotheses_forest.png`
- `plots/calibration.png`
- `plots/timeline_seed00.png`

## Caveats

- This is a synthetic pilot. The data-generating process was designed so that corruption of the observed adjacency is, in principle, visible to structural diagnostics, while the no-signal regime is structurally indistinguishable from the informative one. Any advantage of structural conditioning is therefore conditional on that design and is not evidence about real markets.
- Experts are frozen after expert-val; no re-training through time.
- n = 10 seeds; Wilcoxon's smallest attainable two-sided p is ≈0.002, so Holm-adjusted significance requires near-unanimous sign agreement across seeds.
- No claim of novelty or of guaranteed robustness is made; results describe this benchmark only.
