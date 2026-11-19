# Synthetic pilot — corrected validity run

Final pilot run: `outputs/da_synthetic_pilot_validity_v2/`, config hash `ead62067535d`, seeds
0–9. The earlier run `da_synthetic_pilot_8d3d88171378` remains a diagnostic reference only: its
DGP signal scaling used the complete simulated path, its placebo selector received diagnostics
from the real adjacency, and its old HA6 mixed expert quality with selector contribution.

The corrected run uses pre-sample burn-in scaling, adjacency-matched diagnostics, exact
seed/config aggregation guards, and separates real-vs-placebo expert quality (HD7) from A2a's
incremental selector contribution (HA6).

## Confirmatory verdicts

| ID | Hypothesis | Mean diff [95% CI] | Verdict |
|---|---|---|---|
| HD1 | D1 > 0 in informative regime | +0.270 [0.217, 0.324] | supported |
| HD2 | D1 < 0 in no-signal regime | −0.114 [−0.168, −0.060] | supported |
| HD3 | D1 lower under corruption than informative | −0.250 [−0.315, −0.184] | supported |
| HD4 | D2 > 0 in informative regime | +0.266 [0.219, 0.312] | supported |
| HD5 | D2 lower under corruption than informative | −0.246 [−0.304, −0.188] | supported |
| HD6 | GCN beats cross-asset control in informative regime | +0.190 [0.145, 0.234] | supported |
| HD7 | Real GCN risk lower than retrained placebo GCN | +0.082 [0.045, 0.120] | supported |
| HA1 | A2a regret < A1 | +0.0078 [−0.0040, 0.0197] | inconclusive |
| HA2 | A2a < A2a without structural diagnostics | −0.0015 [−0.0118, 0.0088] | inconclusive |
| HA3 | A2a < A2a without D2 auxiliary | −0.0023 [−0.0086, 0.0040] | inconclusive |
| HA4 | A2a < generic Gate with matched expert/input information | +0.0148 [−0.0020, 0.0316] | inconclusive |
| HA5 | A2a < ConditionalRisk with matched expert/input information | −0.0091 [−0.0271, 0.0089] | inconclusive |
| HA6 | A2a gain over AlwaysG is larger for real than placebo graph | +0.0066 [−0.0136, 0.0267] | inconclusive |
| HA7 | A2b vs A2a (exploratory) | +0.0230 [0.0022, 0.0438] | difference detected; A2a better |

Verdicts use paired seed-level two-sided Wilcoxon tests, Holm correction within the D/A
families, and require the 95% t-interval to exclude zero in the predicted direction.

## Interpretation

The D-side measurement protocol remains strongly supported after the validity fixes. Graph value
is regime-dependent and specific to the real node mapping; this is no longer an artefact of
full-path normalization or mismatched placebo diagnostics.

The corrected run still does not establish a special advantage for A2a. Mean all-regime regret
is 0.0331 for A2a, 0.0183 for Gate, 0.0422 for ConditionalRisk, 0.0253 for A1, and 0.0368 for a
fixed ensemble. A2a strongly beats AlwaysN (difference −0.0895, CI [−0.1233, −0.0557]), but it
does not clearly beat AlwaysG or the fixed ensemble. That result shows value in access to the real
GCN, not a confirmed benefit of A2a's structural-conditioning mechanism.

## Post-hoc diagnostics

These diagnostics reuse the frozen v2 artefacts and are not confirmatory evidence:

- Soft weighting derived from the exact same A2a benefit forecast has risk −0.0069 relative to
  hard switching (95% CI [−0.0183, 0.0045], Wilcoxon p = 0.160). The point estimate favours soft
  weighting, but the comparison is inconclusive with 10 seeds.
- Adding structural columns to an otherwise identical deterministic Ridge benefit estimator
  changes risk by +0.0100 (95% CI [−0.0082, 0.0281]). Nine of ten seeds favour the non-structural
  model, but the effect-size interval crosses zero. There is no evidence that the structural
  diagnostics add decision information in this DGP.

Full primary results are in `outputs/da_synthetic_pilot_validity_v2/report.md`; diagnostic details
are in `diagnostics.json` and `diagnostics_per_seed.csv` in the same directory.

## Predictability ceiling decision

The subsequent ceiling analysis finds that historical evidence is meaningfully predictive of true
conditional graph benefit (test correlation about 0.62–0.65), while structural-only features are
weak (about 0.10–0.15 with negative R²). The dominant gap is between the best observable-feature
oracle-supervised hard selector and exact-benefit hard selection: +0.0290 risk [0.0212, 0.0368].
The noisy-label and current-estimator gaps are smaller and inconclusive, while exact convex
weighting improves on exact hard selection by 0.0056 [0.0035, 0.0077].

This supports proceeding with a **D + A revised / Decision-Aware Graph Value** research stage, but
only to search for new pre-decision reliability evidence and validate it independently. It does not
support making A2a the main method or adding selector complexity. Full details are in
`outputs/da_synthetic_pilot_validity_v2/CEILING_ANALYSIS.md`.

## Stage 9 evidence discovery outcome

The four pre-decision evidence families subsequently screened in Stage 9 did not produce a
Holm-corrected oracle-supervised decision-risk improvement. F1 local residual reliability and F2
conditional error covariance were not supported; F3 graph-message influence and F4 disagreement
distribution were inconclusive. Because no family passed the oracle gate, the independent
realised-supervision holdout was not opened.

The resulting decision is **NO-GO for a revised D + A proposed method** and a pivot to the
Conditional Graph Value Benchmark direction. See `EVIDENCE_DISCOVERY.md` for the full controlled
ablation, per-regime uncertainty, and artefact locations.
