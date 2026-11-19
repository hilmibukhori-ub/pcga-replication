# Conditional graph-benefit predictability ceiling

Status: post-hoc synthetic diagnostic, not primary or confirmatory evidence. All feature rows are
causal and frozen from validity run v2. Model selection uses selector-validation, affine
calibration uses the calibration split, and the locked test is evaluation-only.

Two supervision settings separate available information from noisy estimation:

- **Oracle supervision:** the diagnostic learner receives hidden conditional benefit `B*` as its
  pre-test training target. This is impossible in deployment but estimates how much signal the
  current observable features can carry.
- **Realised supervision:** the same learner receives realised loss differences, matching the
  noisy target available to an implementable selector.

The flexible learner is a small single-process gradient-boosted regressor used only as a ceiling
probe. It is not a proposed replacement for A2a.

## Predictability of true conditional benefit

Mean locked-test results over 10 seeds:

| Supervision / estimator | Features | Corr with B* | Test R² | Hard-selector risk |
|---|---|---:|---:|---:|
| Oracle / Ridge | structural only | 0.142 | −0.225 | 0.327 |
| Oracle / Ridge | historical evidence | 0.644 | 0.240 | 0.297 |
| Oracle / Ridge | decision features, no structure | 0.646 | 0.284 | 0.296 |
| Oracle / Ridge | all | 0.646 | 0.213 | 0.299 |
| Oracle / flexible | structural only | 0.128 | −0.226 | 0.325 |
| Oracle / flexible | historical evidence | 0.619 | 0.225 | 0.299 |
| Oracle / flexible | decision features, no structure | 0.654 | 0.226 | 0.300 |
| Oracle / flexible | all | 0.697 | 0.305 | 0.298 |
| Realised / Ridge | structural only | 0.146 | −0.228 | 0.335 |
| Realised / Ridge | historical evidence | 0.631 | 0.267 | 0.296 |
| Realised / Ridge | decision features, no structure | 0.624 | 0.322 | 0.294 |
| Realised / Ridge | all | 0.616 | 0.254 | 0.304 |
| Realised / flexible | structural only | 0.100 | −0.291 | 0.338 |
| Realised / flexible | historical evidence | 0.574 | 0.193 | 0.304 |
| Realised / flexible | decision features, no structure | 0.501 | −0.020 | 0.310 |
| Realised / flexible | all | 0.515 | −0.020 | 0.305 |

Historical evidence beats structural-only risk by roughly 0.026–0.038, depending on supervision
and estimator. With oracle supervision, adding structure to all non-structural decision features
improves the flexible learner by only 0.0022 risk [0.0001, 0.0042] and raises R² by 0.079. Under
realised supervision that marginal benefit is not robust. Structure therefore contains weak
complementary information in this DGP, but the implementable learning signal does not reliably
extract it.

## Risk ladder and theoretical value

| Decision rule | Mean conditional risk |
|---|---:|
| Always N | 0.392 |
| Always G | 0.314 |
| Fixed ensemble | 0.306 |
| A2a hard | 0.302 |
| A2a soft, same forecast | 0.295 |
| Oracle-supervised observable-feature learner, hard | 0.298 |
| Realised-supervised observable-feature learner, hard | 0.305 |
| Generic Gate | 0.287 |
| Exact conditional benefit, hard | 0.269 |
| Exact optimal convex mixture | 0.263 |

Knowing exact conditional benefit would reduce risk substantially: exact hard selection gains
0.033 relative to A2a hard. Exact convex weighting adds another 0.0056 [0.0035, 0.0077] beyond
exact hard selection.

## Gap decomposition

| Gap | Mean [95% CI] | Interpretation |
|---|---|---|
| Empirical information/representation | +0.0290 [0.0212, 0.0368] | Observable all-feature oracle learner minus exact-benefit hard oracle |
| Noisy-label estimation | +0.0067 [−0.0059, 0.0193] | Realised-supervised minus oracle-supervised feature learner |
| Current-estimator | −0.0026 [−0.0194, 0.0141] | A2a hard minus realised-supervised flexible learner |
| Hard-to-soft, same A2a forecast | −0.0069 [−0.0183, 0.0045] | Negative favours soft; inconclusive |
| Exact soft-oracle advantage | −0.0056 [−0.0077, −0.0035] | Convex oracle minus hard oracle |

The dominant measured bottleneck is the information/representation gap. There is no evidence that
A2a is mainly failing because its estimator is too simple. Soft decisions have some theoretical
and empirical value, but their scale is smaller than the missing-information gap.

## Decision for the next research stage

Proceed with **D + A revised / Decision-Aware Graph Value**, subject to an independent validation
stage. Do not promote A2a or the diagnostic flexible learner as the proposed method.

The next feature study should prioritize pre-decision predictive evidence, for example local
expert residual reliability, cross-sectional disagreement distributions, graph-message influence
stability, and recent conditional error covariance. New signals should pass two gates before any
selector architecture work:

1. Under oracle supervision, they must reduce hard decision risk beyond the existing
   evidence+disagreement+market feature set.
2. Under realised supervision, the incremental gain must survive on independent DGP settings or
   untouched seeds.

If those gates fail, further selector complexity is unlikely to close the oracle gap and the work
should revert to the Conditional Graph Value Benchmark direction.
