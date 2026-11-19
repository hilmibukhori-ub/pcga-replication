# PCGA-L construct-recovery report

**Status:** post-hoc method packaging of a previously preregistered experiment. 
The PCGA-L name and this report were not preregistered; the scenarios, ladder, 
thresholds, hypotheses, seeds and stored test losses were fixed in the earlier 
`placebo_ladder_v1` study. No model was fitted by this report.

## What was verified

The loss decomposition is `D1 = density + degree + block + edge`. Steps may be 
negative; monotonic improvement is not assumed. The maximum absolute telescoping 
error in the stored results is `0.000e+00`.

## Recovery of the known signal level

| scenario | expected | dominant | expected positive-value share | seeds >= 70% | recovery rule |
|---|---|---|---:|---:|---|
| edge | edge | edge | 0.915 | 100% | met |
| block | block | block | 0.907 | 95% | met |
| degree | degree | degree | 0.967 | 100% | met |

All three signal-bearing scenarios recover the structural level that generated 
the signal under the old preregistered 70%/80% rule. The null scenario remains a 
negative control and is assessed by the old H6 tests, not by forcing a dominant step.

## Null negative control

| step | mean loss contribution | old Holm-adjusted p | old verdict |
|---|---:|---:|---|
| density | -1.605e-05 | 0.278 | no clear difference |
| degree | -1.325e-05 | 0.861 | no clear difference |
| block | +1.230e-05 | 0.736 | no clear difference |
| edge | -5.416e-06 | 0.898 | no clear difference |

## Claim boundary

This supports construct recovery for the transparent ridge data-generating process. 
It is not evidence that market relations are causal, that the ladder is universally 
identified, or that every graph model will recover the same level. Empirical relation 
specificity still requires the pre-specified constrained-placebo test, and practical 
value still requires an independently chosen materiality threshold.
