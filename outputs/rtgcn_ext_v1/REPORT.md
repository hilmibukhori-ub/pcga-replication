# RT-GCN extension — results

Holm family complete: True (6 of 6 tests).

- **H-A_NYSE_U**: +3.324 [95% CI -1.471, +8.119], p 0.0815, Holm 0.4893 → not detected (9/20 seeds positive)
- **H-B_NYSE_U**: -1.791 [95% CI -5.416, +1.835], p 0.8429, Holm 1.0000 → not detected (8/20 seeds positive)
- **H-A_NASDAQ_T**: -0.093 [95% CI -0.856, +0.670], p 0.6055, Holm 1.0000 → not detected (4/10 seeds positive)
- **H-B_NASDAQ_T**: -0.155 [95% CI -0.488, +0.178], p 0.8405, Holm 1.0000 → not detected (2/10 seeds positive)
- **H-A_NYSE_T**: +3.307 [95% CI -4.241, +10.854], p 0.1738, Holm 0.7069 → not detected (4/10 seeds positive)
- **H-B_NYSE_T**: +3.864 [95% CI -3.787, +11.515], p 0.1414, Holm 0.7069 → not detected (7/10 seeds positive)
- **H-C NYSE_U** (exact rank, K=19, unadjusted): real -0.006 vs placebo mean +0.212; 15/19 ≥ real; p = 0.800
- R0 NYSE_U: irr1 0.605 [-0.174, 1.384] (paper 0.65), irr5 0.666 [0.330, 1.003] (paper 0.88), irr10 0.693 [0.568, 0.818] (paper 0.96, GAP), mrr 0.041 [0.031, 0.051] (paper 0.04)
- variance NYSE_U: within_run_sd_over_epochs_mean 4.416, between_seed_sd_at_val_epoch 7.467, corr_testloss_IRR1_mean 0.107, oracle_best_epoch_IRR1_mean 16.912
- **H-C NASDAQ_T** (exact rank, K=19, unadjusted): real +0.273 vs placebo mean +0.182; 9/19 ≥ real; p = 0.500
- R0 NASDAQ_T: irr1 -0.055 [-0.675, 0.565] (paper 1.25, GAP), irr5 0.615 [0.211, 1.020] (paper 0.97), irr10 0.670 [0.391, 0.949] (paper 1.03, GAP), mrr 0.041 [0.033, 0.049] (paper 0.061, GAP)
- variance NASDAQ_T: within_run_sd_over_epochs_mean 0.551, between_seed_sd_at_val_epoch 0.688, corr_testloss_IRR1_mean -0.135, oracle_best_epoch_IRR1_mean 1.754
- **H-C NYSE_T** (exact rank, K=19, unadjusted): real +2.967 vs placebo mean +2.810; 7/19 ≥ real; p = 0.400
- R0 NYSE_T: irr1 0.578 [-0.836, 1.992] (paper 0.92), irr5 0.471 [0.233, 0.710] (paper 1.1, GAP), irr10 0.656 [0.523, 0.789] (paper 1.13, GAP), mrr 0.038 [0.029, 0.047] (paper 0.056, GAP)
- variance NYSE_T: within_run_sd_over_epochs_mean 7.907, between_seed_sd_at_val_epoch 10.349, corr_testloss_IRR1_mean -0.072, oracle_best_epoch_IRR1_mean 32.203