# Results — RSR open-track graph-value ladder (confirmatory)

**Draft v0.1, 2026-11-10.** Pre-registration: `PREREG_RSR.md` (frozen 2026-11-09, `PREREG_FREEZE.md`).
Deviations: `DEVIATIONS.md` (C1, C2, P1, P1b). Raw output: `outputs/rsr_ladder_v1/results.json`,
`logs/rsr_confirm.log`. Run started 2026-11-10 04:39 CET. It was a single read of the test
period, and the freeze-hash check passed.

All results are fixed-universe historical results on the RSR universe (track O). They are not
market-wide claims.

## 1. Run record

| | NASDAQ | NYSE |
|---|---|---|
| Test period (after purge) | 2017-01-03 → 2017-12-07, 236 days | same |
| Ridge α, L0 / L4 (validation) | 1 / 1e4 | **1e6 / 1e6** (grid edge) |
| L3 placebos | 99, all reported | 99, all reported |
| Mean edge overlap L3 ↔ Wiki | 0.55 | 0.60 |
| Winsorisation (train 0.5/99.5%) | [−0.061, 0.068] | [−0.049, 0.051] |
| Test MSE, L4 (winsorised) | 2.3487e-4 | 1.4749e-4 |

## 2. Confirmatory results (R1–R4, Holm across 4 tests, α = 0.05)

Quantities are paired differences in mean daily test MSE (winsorised target). A positive value
means the graph model has the lower loss. D1 = L0 − L4. edge = mean(L3) − L4.

| ID | Market | Quantity | Estimate | Relative to L4 MSE | 90% interval | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|---|---|
| R1 | NASDAQ | D1 | +7.1e-9 | +0.003% | [−3.0e-8, +5.2e-8] (bootstrap) | 0.413 | 0.413 | not detected |
| R2 | NYSE | D1 | +2.5e-9 | +0.002% | [−2.2e-9, +7.0e-9] (bootstrap) | 0.193 | 0.386 | not detected |
| R3 | NASDAQ | edge | +1.5e-8 | +0.006% | [−5.6e-9, +3.7e-8] (bootstrap, descriptive) | 0.040 | 0.120 | not detected ‡ |
| R4 | NYSE | edge | +4.0e-9 | +0.003% | [−8.2e-10, +8.7e-9] (bootstrap, descriptive) | 0.010 | 0.040 | **supported** |

D1 inference uses a circular block bootstrap (L = 20, 1,000 repetitions). Edge inference uses the
placebo-rank test, `p = (1 + #{k : loss(L3_k) ≤ loss(L4)}) / 100`. Under §6.2, the bootstrap
interval for edge is descriptive only (its size in the power study was 0.14–0.15), so the edge
verdicts rest on the rank p-value. For R4, the Wiki graph has a lower test loss than all 99
placebos, which gives p = 0.01, the smallest p-value possible at K = 99. For R3, 3 of the 99
placebos have a lower loss than Wiki.

‡ **Calibration label (DEVIATIONS P1b, fixed before the test read):** on the real NASDAQ Wiki
topology, the edge test's simulated size was 20/300 = 0.067 (Wilson 95% upper bound 0.101 > 0.08).
R3 is therefore labelled **"calibration doubtful on this topology"**. R3 is not significant
after Holm in any case, so the label does not change the verdict. NYSE edge (2/100, upper bound
0.070) and D1 in both markets (upper bounds ≤ 0.056) pass the calibration check.

## 3. Pre-registered interpretation (§8, applied verbatim)

R4 is supported while R2 is not. The pre-registration fixed in advance how to read this pattern:
the real edges beat matched placebos even though the graph as a whole does not beat no graph.

In words: on NYSE, the specific Wiki edges carry predictive information beyond what
degree-preserving and industry-composition-preserving placebo graphs provide, when each graph
is used in a retrained linear expert. Whether access to the Wiki graph lowers loss compared with
using no graph is **not detected** on either market.

**Look-ahead caveat (§4a, pre-registered):** the Wiki relations were collected after the sample
period and are not point-in-time. A positive edge result is exactly the case where this caveat
applies. R4 may partly reflect relations that became known only after 2017. The null results
are not affected by this.

## 4. Effect size and power context

- **Every effect is economically tiny.** The largest point estimate is a 0.006% reduction in test
  MSE. Daily Pearson IC on NYSE is 0.0139 for L0 and 0.0149 for L4. On NASDAQ it is 0.0131 for L0
  and 0.0137 for L4.
- **Power context for the nulls (P1, real Wiki topology, no RSR returns):** MDE80 at the
  Holm-worst level is R² ≈ 2e-4 (IC ≈ 0.016) for D1 and R² ≈ 1e-4 (IC ≈ 0.011) for edge, in both
  markets. These figures are optimistic upper bounds, because the simulated ridge knows the true
  functional form. Under §8, "not detected" for R1–R3 means no effect at or above roughly this
  scale was found. It is not evidence of zero effect or of equivalence.
- **Why R4 is significant despite its small size:** the 99 NYSE placebo losses are very tightly
  clustered (range 1.474930e-4 to 1.474960e-4, a spread of 3.0e-9). One likely reason is that
  α = 1e6 shrinks every model strongly. Wiki's loss (1.474906e-4) falls below the whole cluster.
  The rank test is exact under exchangeability, so this is a valid rejection, but it is a
  rejection of an extremely small effect.

## 5. Exploratory: ladder decomposition (§9.1)

*Exploratory. These are point estimates only; the bootstrap intervals for each step have not been
computed yet. They are not confirmatory evidence.*

The four steps sum exactly to D1: density + degree + industry + edge.

| Step (Δ MSE, + = lower loss) | NASDAQ | NYSE |
|---|---|---|
| density (L0 → L1) | −5.4e-9 | +0.3e-9 |
| degree (L1 → L2) | −10.9e-9 | −2.9e-9 |
| industry (L2 → L3) | +8.2e-9 | +1.1e-9 |
| edge (L3 → L4) | +15.2e-9 | +4.0e-9 |
| **D1 (sum)** | +7.1e-9 | +2.5e-9 |

In both markets the degree step has the most negative point estimate, and the edge step has the
largest positive one. One reading is that degree-matched random neighbours add noise that offsets
part of the Wiki edge contribution, so the net D1 ends up smaller than edge. **This reading was
formed after seeing the results.** It is a hypothesis for the paper's discussion section and for
future pre-registered work. It is not a finding. The per-step intervals (§9.1) are needed before
it can be described further.

## 6. Exploratory: secondary losses (§9.5)

| | NASDAQ D1 | NASDAQ edge | NYSE D1 | NYSE edge |
|---|---|---|---|---|
| MSE, unwinsorised | +8.4e-9 | +1.4e-8 | +4.0e-9 | +5.9e-9 |
| MAE | −4.4e-7 | +1.6e-7 | −5.7e-8 | +3.0e-8 |

On NYSE, the edge sign agrees across all three losses. Wiki has the lowest unwinsorised MSE of
any graph, beating all 99 placebos. On MAE it is better than the placebo median but not the best.
D1 changes sign under MAE in both markets.

## 6b. Exploratory: GCN/MLP robustness (§5, §9.4; DEVIATIONS G1)

*Exploratory and descriptive. The spec was fixed after the confirmatory test read (G1). Matched
2-layer GCN vs MLP with 3 seeds, losses averaged over seeds. The edge test uses the first 19 of
the 99 confirmatory L3 placebos (K = 19, so the minimum p is 0.05). Full output:
`outputs/rsr_gnn_v1/REPORT.md`.*

| | NASDAQ D1 | NASDAQ edge | NYSE D1 | NYSE edge |
|---|---|---|---|---|
| **Ridge (confirmatory)** | +7.1e-9 (p .41) | +1.5e-8 (p .04; 3/99 better) | +2.5e-9 (p .19) | **+4.0e-9 (p .01; 0/99 better)** |
| **GCN (exploratory)** | +2.1e-8 (p .16) | **+2.3e-8 (p .05; 0/19 better)** | −2.7e-8 (90% CI below 0) | −1.4e-8 (17/19 better) |
| GCN seeds positive | 3/3 | 3/3 | 1/3 | 1/3 |

**The edge result does not replicate across expert classes. The direction reverses between
markets.**

- **NASDAQ:** GCN-Wiki beats all 19 placebos in every seed, and the bootstrap interval excludes
  zero. This is the reverse of the ridge result, where R3 was not detected and its calibration
  is doubtful.
- **NYSE:** GCN-Wiki loses to most placebos and to the MLP. However, this result is dominated by
  an optimisation failure. In seeds 0 and 1, GCN-Wiki was early-stopped at epochs 1 and 4, so the
  model was essentially untrained, and in 10 of the 81 NYSE fits the best epoch was ≤ 2. The one
  seed that trained for long (epoch 52) has positive D1 and edge. The validation loss on NYSE is
  too flat for the early-stopping rule to work reliably. The NYSE GCN result is therefore weak
  evidence in either direction. It does not contradict R4, but it does not support it either.
- **The ladder pattern in §5 does not replicate.** Under the GCN, the degree step is positive in
  both markets (+6.5e-9 and +8.8e-9). The idea that the degree step masks the edge gain is
  therefore not supported, and it should not appear in the paper as more than a ridge-specific
  observation.

**Implication:** for this data, a claim about which edges matter depends on both the expert and
the market. That supports the paper's methodological argument, that a single-model graph claim
is fragile and should be tested against retrained placebos. It weakens any substantive claim
that the Wiki relations carry robust predictive value. A sensitivity run with a minimum training
length would address the NYSE early-stopping failure, but it would be a post-hoc choice and has
not been run.

## 7. Limitations

Pre-registered (§11), plus items that arose during the run:

1. Fixed RSR universe, likely survivorship. One market history, one test year (236 days).
2. Wiki and industry relations are not point-in-time. This matters most for R4 (§3).
3. **NYSE ridge chose α = 1e6, the edge of the frozen grid, for both L0 and L4** (this was
   already seen in the dev smoke test). The grid was not changed. The NYSE results therefore
   describe a heavily shrunk linear expert.
4. The linear expert only. A null result does not rule out nonlinear graph value. The GCN/MLP
   robustness check (§6b) does not replicate the ridge edge pattern.
5. Wiki is 85–87% isolated nodes, so any graph effect is concentrated in the 13–15% of stocks
   that have neighbours. Panel-wide effect sizes understate the effect on connected stocks.
6. Industry label 'n/a' (156 of 500 stocks in the intake) was treated as a singleton per stock
   (C2).
7. The edge test's calibration on the NASDAQ topology is doubtful (P1b, §2).

## 8. Not yet run (pre-registered secondary analyses, §9)

- [ ] Bootstrap intervals for each ladder step (§9.1)
- [ ] Pooled market (§9.2)
- [ ] Train-period correlation graph as L4 (§9.3)
- [x] GCN/MLP robustness, 3 seeds, K = 19 (§9.4); see §6b
- [ ] Rank IC (§9.5)
- [ ] Conditional table by volatility and dispersion terciles (§9.6)
- [ ] Block-length sensitivity, L = 40 and 60 (§9.7)

These analyses all use the test period, which has already been read. They are exploratory by
design, and running them does not reopen the confirmatory family.
