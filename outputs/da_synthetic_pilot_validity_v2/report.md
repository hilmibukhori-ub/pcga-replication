# D + A synthetic pilot — results report

Run: `da_synthetic_pilot_validity_v2` · config hash `ead62067535d` · seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

All numbers are locked-test (time steps 2001–3000) and aggregated over seeds; intervals are 95% t-intervals over seeds (n = 10). Hypothesis verdicts use a two-sided Wilcoxon signed-rank test on per-seed paired differences, Holm-adjusted within each family (D, A), plus the requirement that the t-interval excludes zero in the stated direction.

## Hypotheses

| ID | Hypothesis | Mean diff | 95% CI | Wilcoxon p (Holm) | +/− seeds | Verdict |
|---|---|---|---|---|---|---|
| HD1 | Graph value D1 (GCN vs matched MLP) > 0 in informative regime | 0.2703 | [0.2169, 0.3237] | 0.0137 | 10/0 | **supported** |
| HD2 | D1 < 0 in no-signal regime (graph hurts without relational signal) | -0.1143 | [-0.1683, -0.0604] | 0.0137 | 0/10 | **supported** |
| HD3 | D1 lower under corrupted adjacency than informative | -0.2497 | [-0.3153, -0.1841] | 0.0137 | 0/10 | **supported** |
| HD4 | D2 (placebo minus real graph) > 0 in informative regime | 0.2657 | [0.2192, 0.3121] | 0.0137 | 10/0 | **supported** |
| HD5 | D2 lower under corrupted adjacency than informative | -0.2459 | [-0.3038, -0.1880] | 0.0137 | 0/10 | **supported** |
| HD6 | GCN beats cross-asset non-graph control in informative regime | 0.1896 | [0.1451, 0.2341] | 0.0137 | 10/0 | **supported** |
| HD7 | Real GCN has lower conditional risk than a retrained placebo GCN | 0.0823 | [0.0450, 0.1197] | 0.0137 | 10/0 | **supported** |
| HA1 | A2a regret < A1 regret | 0.0078 | [-0.0040, 0.0197] | 0.0586 | 8/2 | **inconclusive** |
| HA2 | A2a regret < A2a without structural diagnostics | -0.0015 | [-0.0118, 0.0088] | 1.0000 | 6/4 | **inconclusive** |
| HA3 | A2a regret < A2a without D2 auxiliary task | -0.0023 | [-0.0086, 0.0040] | 1.0000 | 5/5 | **inconclusive** |
| HA4 | A2a regret < generic gate (same features) | 0.0148 | [-0.0020, 0.0316] | 0.0977 | 9/1 | **inconclusive** |
| HA5 | A2a regret < conditional-risk selector (same features) | -0.0091 | [-0.0271, 0.0089] | 0.3359 | 1/9 | **inconclusive** |
| HA6 | A2a's gain over AlwaysG is larger with the real graph than with placebo | 0.0066 | [-0.0136, 0.0267] | 1.0000 | 5/5 | **inconclusive** |
| HA7 | A2b (dual evidence) vs A2a regret [exploratory, two-sided] | 0.0230 | [0.0022, 0.0438] | 0.0488 | 7/3 | **difference detected** |

Sign convention: D rows are the quantity itself (or a regime difference); A rows are regret(A) − regret(B), so negative favours the first-named method. HD7 isolates the raw real-vs-placebo expert-quality gap. HA6 is a difference-in-differences: A2a's conditional-risk gain over AlwaysG in the real slot minus that gain in the placebo slot.

## Selector regret by regime (real graph slot)

| Method | all | informative | no_signal | corrupted | graph weight (all) | oracle agreement |
|---|---|---|---|---|---|---|
| AlwaysN | 0.1226 ± 0.0309 | 0.2691 ± 0.0480 | 0.0016 ± 0.0015 | 0.0587 ± 0.0096 | 0.00 | 0.43 |
| AlwaysG | 0.0451 ± 0.0164 | 0.0001 ± 0.0001 | 0.1159 ± 0.0426 | 0.0333 ± 0.0148 | 1.00 | 0.57 |
| FixedEnsemble | 0.0368 ± 0.0076 | 0.0282 ± 0.0146 | 0.0712 ± 0.0200 | 0.0128 ± 0.0044 | 0.83 | 0.59 |
| Gate | 0.0183 ± 0.0056 | 0.0253 ± 0.0150 | 0.0203 ± 0.0106 | 0.0101 ± 0.0040 | 0.65 | – |
| CondRisk | 0.0422 ± 0.0081 | 0.0410 ± 0.0199 | 0.0484 ± 0.0248 | 0.0417 ± 0.0096 | 0.64 | 0.64 |
| A1 | 0.0253 ± 0.0067 | 0.0138 ± 0.0077 | 0.0309 ± 0.0191 | 0.0351 ± 0.0096 | 0.61 | 0.75 |
| A2a | 0.0331 ± 0.0175 | 0.0332 ± 0.0437 | 0.0311 ± 0.0166 | 0.0357 ± 0.0116 | 0.57 | 0.69 |
| A2a_noStruct | 0.0346 ± 0.0152 | 0.0424 ± 0.0376 | 0.0252 ± 0.0126 | 0.0376 ± 0.0084 | 0.54 | 0.70 |
| A2a_noAux | 0.0354 ± 0.0225 | 0.0363 ± 0.0551 | 0.0311 ± 0.0207 | 0.0372 ± 0.0119 | 0.57 | 0.68 |
| A2b | 0.0561 ± 0.0275 | 0.0923 ± 0.0771 | 0.0273 ± 0.0217 | 0.0423 ± 0.0127 | 0.44 | 0.62 |
| Oracle | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.57 | 1.00 |

## Graph value (D1, D2) by regime

| Quantity | all | informative | no_signal | corrupted |
|---|---|---|---|---|
| D1_realised_vs_mlp | 0.0755 [0.038, 0.113] | 0.2703 [0.217, 0.324] | -0.1143 [-0.168, -0.060] | 0.0206 [-0.011, 0.052] |
| D1_oracle_vs_mlp | 0.0775 [0.046, 0.109] | 0.2690 [0.221, 0.317] | -0.1143 [-0.157, -0.071] | 0.0255 [0.003, 0.048] |
| D1_realised_vs_cross_mlp | 0.0479 [0.002, 0.094] | 0.1896 [0.145, 0.234] | -0.0284 [-0.073, 0.016] | -0.0399 [-0.107, 0.027] |
| D1_realised_vs_ridge | 0.1850 [0.116, 0.254] | 0.3782 [0.296, 0.460] | -0.0064 [-0.081, 0.069] | 0.1395 [0.078, 0.201] |
| D1placebo_realised_vs_mlp | -0.0056 [-0.018, 0.007] | 0.0046 [-0.012, 0.021] | -0.0267 [-0.042, -0.011] | 0.0008 [-0.010, 0.012] |
| D2_realised | 0.0811 [0.043, 0.119] | 0.2657 [0.219, 0.312] | -0.0877 [-0.132, -0.043] | 0.0198 [-0.016, 0.056] |
| D2_oracle | 0.0807 [0.048, 0.114] | 0.2614 [0.220, 0.302] | -0.0892 [-0.127, -0.051] | 0.0229 [-0.005, 0.051] |

## Graph-value estimation error and calibration (real slot, all test steps)

| Method | benefit RMSE vs B* | corr with B* | calib. slope (median) | calib. slope (mean) | calib. intercept (median) | binned calib. error | D2 RMSE |
|---|---|---|---|---|---|---|---|
| CondRisk | 0.216 | 0.352 | 0.466 | 0.423 | 0.045 | 0.091 | – |
| A1 | 0.162 | 0.624 | 1.048 | 1.068 | 0.031 | 0.046 | – |
| A2a | 0.175 | 0.617 | 1.176 | 1.890 | 0.046 | 0.070 | 0.142 |
| A2a_noStruct | 0.172 | 0.558 | 0.941 | 1.149 | 0.044 | 0.057 | 0.142 |
| A2a_noAux | 0.178 | 0.624 | 1.303 | 3.402 | 0.042 | 0.074 | – |

Calibration slope = OLS slope of true benefit B* on the calibrated prediction (ideal 1). The seed mean is unstable: in seeds where a selector's test-period predictions are nearly constant the slope explodes (per-seed values in `per_seed_results.csv`), so the median is the robust summary.

## Ablation summary (regret A − B, real slot)

| Comparison | all | informative | no_signal | corrupted |
|---|---|---|---|---|
| A1 vs A2a | 0.0078 [-0.0040, 0.0197] | 0.0194 [-0.0213, 0.0600] | 0.0003 [-0.0233, 0.0238] | 0.0006 [-0.0049, 0.0061] |
| A2a: with vs without structural diagnostics | -0.0015 [-0.0118, 0.0088] | -0.0092 [-0.0399, 0.0215] | 0.0059 [-0.0045, 0.0164] | -0.0020 [-0.0094, 0.0055] |
| A2a: with vs without D2 auxiliary | -0.0023 [-0.0086, 0.0040] | -0.0032 [-0.0151, 0.0088] | 0.0001 [-0.0127, 0.0128] | -0.0015 [-0.0067, 0.0036] |
| A2a vs generic gate | 0.0148 [-0.0020, 0.0316] | 0.0079 [-0.0405, 0.0563] | 0.0108 [-0.0071, 0.0287] | 0.0256 [0.0125, 0.0387] |
| A2a vs conditional-risk | -0.0091 [-0.0271, 0.0089] | -0.0078 [-0.0601, 0.0445] | -0.0172 [-0.0399, 0.0054] | -0.0061 [-0.0126, 0.0005] |
| A2b vs A2a | 0.0230 [0.0022, 0.0438] | 0.0591 [-0.0001, 0.1182] | -0.0038 [-0.0155, 0.0078] | 0.0066 [-0.0020, 0.0152] |
| A1 vs A2a_noStruct (model class only) | 0.0093 [-0.0013, 0.0199] | 0.0286 [-0.0054, 0.0625] | -0.0057 [-0.0224, 0.0111] | 0.0025 [-0.0020, 0.0071] |
| A2a vs AlwaysN | -0.0895 [-0.1233, -0.0557] | -0.2359 [-0.3005, -0.1713] | 0.0296 [0.0127, 0.0464] | -0.0231 [-0.0396, -0.0065] |
| A2a vs AlwaysG | -0.0120 [-0.0292, 0.0052] | 0.0331 [-0.0106, 0.0768] | -0.0847 [-0.1306, -0.0388] | 0.0024 [-0.0100, 0.0148] |
| A2a vs FixedEnsemble | -0.0037 [-0.0183, 0.0109] | 0.0049 [-0.0408, 0.0507] | -0.0401 [-0.0680, -0.0122] | 0.0229 [0.0124, 0.0335] |

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
