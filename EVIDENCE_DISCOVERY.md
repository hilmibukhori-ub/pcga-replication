# Stage 9 — Decision-Relevant Evidence Discovery

## Status and scope

Stage 9 preserves the corrected v2 DGP, chronological splits, label-maturity rules, frozen graph
and non-graph experts, and conditional-risk evaluation. It does not modify the locked v2 results.
The oracle screen is a post-hoc synthetic discovery analysis, not confirmatory evidence and not a
claim of novelty.

Primary comparison: existing causal decision features (`evidence + disagreement + market`) versus
the exact same features plus one candidate family. Every comparison uses the same oracle-`B*`
training target, gradient-boosted estimator class and hyperparameter budget, splits, calibration,
expert predictions, and hard graph/MLP decision rule. The primary endpoint is locked-test
incremental conditional risk; correlation is secondary. Holm correction is applied across the
four primary all-regime family comparisons.

## Evidence families

| Family | Added features | Construction |
|---|---:|---|
| F1 — local residual reliability | 21 | Per-asset matured rolling graph/non-graph losses and benefit, summarized by cross-sectional quantiles, dispersion, and fraction positive |
| F2 — conditional error covariance | 12 | Matured rolling error variances/covariance, stabilized diversification denominator, unique-error components, and clipped candidate mixture weight |
| F3 — graph-message influence stability | 12 | GCN prediction sensitivity to deterministic 20% current-edge dropout and one-step-lag adjacency; no hidden graph information |
| F4 — disagreement distribution | 16 | Current signed disagreement distribution plus rolling signed and magnitude quantiles; excludes existing MAD/RMS columns |

F1/F2 use only outcomes whose labels have matured by the decision time. F3 reconstructs the frozen
GCN deterministically, verifies its base predictions against the saved v2 artefact, and perturbs
only current/past observed adjacency. F4 uses current or past frozen predictions only.

## Oracle-supervised controlled ablation

Risk difference is `(existing decision features + family) − existing decision features`; negative
favours the added family. Intervals are paired 95% t-intervals over the ten v2 seeds. `p_holm`
uses paired Wilcoxon tests with Holm correction across F1–F4.

| Family | All-regime risk diff [95% CI] | p_holm | Corr(B*) diff [95% CI] | Screen decision |
|---|---:|---:|---:|---|
| F1 | +0.00738 [−0.01036, +0.02512] | 1.0000 | −0.0260 [−0.0598, +0.0079] | Not supported |
| F2 | +0.00516 [−0.00823, +0.01855] | 1.0000 | +0.0024 [−0.0190, +0.0238] | Not supported |
| F3 | −0.00040 [−0.00086, +0.00006] | 0.1136 | +0.0039 [−0.0027, +0.0104] | Inconclusive |
| F4 | −0.00191 [−0.00682, +0.00300] | 0.9810 | −0.0048 [−0.0551, +0.0455] | Inconclusive |

No candidate passes the pre-specified requirement of negative mean risk, a 95% interval below
zero, and Holm-adjusted p < 0.05.

## Per-regime risk differences

| Family | informative | no_signal | corrupted |
|---|---:|---:|---:|
| F1 | +0.02114 [−0.02033, +0.06260] | −0.00312 [−0.00858, +0.00234] | +0.00023 [−0.00547, +0.00593] |
| F2 | +0.01306 [−0.01847, +0.04458] | −0.00323 [−0.00861, +0.00215] | +0.00137 [−0.00332, +0.00606] |
| F3 | −0.00001 [−0.00206, +0.00204] | −0.00140 [−0.00366, +0.00086] | +0.00024 [−0.00041, +0.00088] |
| F4 | −0.00261 [−0.01436, +0.00915] | −0.00101 [−0.01000, +0.00798] | −0.00164 [−0.00497, +0.00170] |

The small no-signal improvements for F1–F3 are uncertain and do not offset the absence of a
reliable all-regime gain. These non-significant results are not interpreted as equivalence.

## Independent realised-supervision evaluation

Not run. Task 4 is explicitly gated on passing the oracle screen, and no family passed. Opening
new seeds or shifted DGP settings after a failed screen would turn the independent holdout into a
second discovery set. The independent seed/DGP holdout therefore remains untouched.

This means no family can be classified as “supported for further development,” because none has
demonstrated incremental decision value under both oracle and independent realised supervision.

## Research classification

| Family | Classification | Reason |
|---|---|---|
| F1 | Not supported | Mean risk worsens and predictive correlation does not improve |
| F2 | Not supported | Covariance-derived evidence and candidate weights do not reduce oracle-screen risk |
| F3 | Inconclusive | Very small favourable point estimate, but CI crosses zero and Holm p = 0.1136 |
| F4 | Inconclusive | Favourable risk point estimate is small, uncertain, and not accompanied by better B* correlation |

## Decision: NO-GO for revised D + A method

No evidence family meets the required two-stage standard. Do not promote A2a, add selector
complexity, tune these families further on v2, or proceed to real financial data as a method paper.

The recommended research direction is to pivot to a **Conditional Graph Value Benchmark** built
around the validated D1/D2 protocol. F3/F4 may be mentioned only as inconclusive synthetic
diagnostics; any renewed method-development attempt would require a newly pre-specified evidence
hypothesis and a fresh discovery benchmark, not another adjustment against v2.

## Artefacts

- `outputs/stage9_evidence_discovery/oracle_screen.json` — complete machine-readable results.
- `outputs/stage9_evidence_discovery/oracle_screen_per_seed.csv` — per-seed and per-regime rows.
- `outputs/stage9_evidence_discovery/oracle_screen_ablation.csv` — feature-family ablation table.
- `outputs/stage9_evidence_discovery/independent_evaluation_summary.json` — unopened-holdout record.
- `dapilot/evidence.py`, `dapilot/discovery.py` — feature and screening implementation.
- `tests/test_evidence.py` — maturity, causality, non-duplication, stabilization, and screening tests.
