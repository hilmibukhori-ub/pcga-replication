# RT-GCN re-test — results

- **H-A_inflation_IRR1**: estimate +0.284 [95% CI -0.011, +0.580], p 0.0293, Holm 0.0586 → not detected (10/20 seeds positive)
- **H-B_D1_IRR1**: estimate -0.473 [95% CI -1.059, +0.113], p 0.9462, Holm 0.9462 → not detected (7/20 seeds positive)
- **H-C edge** (exact rank, K=19): real +0.869 vs placebo mean +0.932; 8/19 placebos ≥ real; p = 0.450
- **R0 (published protocol, test-selected, day 89 deleted)**: irr1 0.352±1.167 (paper 0.45), irr5 0.813±0.315 (paper 0.71), irr10 0.899±0.210 (paper 0.82), mrr 0.050±0.016 (paper 0.055)
- released_cfg_D1_IRR1 (descriptive): +0.290 [+0.011, +0.570], p 0.0218, 7/10
- released_cfg_inflation_IRR1 (descriptive): +0.136 [-0.167, +0.439], p 0.1687, 3/10
- D1_industry_only_IRR1 (descriptive): -0.226 [-3.158, +2.705], p 0.5797, 1/5
- variance_IRR1: within_run_sd_over_epochs_mean 0.703, between_seed_sd_at_val_epoch 0.788, corr_testloss_IRR1_mean 0.160, oracle_best_epoch_IRR1_mean 2.324