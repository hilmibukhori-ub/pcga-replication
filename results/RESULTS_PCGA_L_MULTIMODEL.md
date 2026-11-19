# PCGA-L cross-model construct recovery: confirmatory results

## Status

Prospective protocol: `PREREG_PCGA_L_MULTIMODEL.md`. Configuration, implementation,
runner, and tests were hash-frozen in `PCGA_L_MULTIMODEL_FREEZE.json` before any
confirmatory seed was run. Development seeds 9000--9001 are excluded. The confirmatory
analysis contains all and only seeds 100--119.

## Integrity checks

- 20 independent DGP seeds and 240 family/scenario rows are complete.
- Every cell contains exactly 20 seeds.
- All 2,640 separately fitted graph-variant models have finite test loss.
- MLP parameter count is 129 at every graph level; GCN parameter count is 913 at every
  graph level. L0 therefore does not receive a smaller architecture.
- L1/L2/L3 contain three independently retrained placebo graphs per seed; L0/L4 contain
  one graph condition each.
- Maximum absolute error in `D1 = density + degree + block + edge` is
  `2.98e-16`.

## Pre-specified verdict

The strict global criterion was **not met**. All nine directional recovery tests were
supported after Holm correction, and the expected step was the largest mean step in all
nine model/scenario cells. However, the per-seed concentration rule required the expected
step to account for at least 70% of summed positive attribution in at least 80% of seeds.
The three block-signal cells reached only 75% (ridge), 65% (MLP), and 65% (GCN).

| Family | Scenario | Expected step | Mean step | 95% t interval | Mean positive-value share | Seeds with share >=70% | Directional test | Concentration |
|---|---|---|---:|---:|---:|---:|---|---|
| Ridge | Edge | Edge | 0.006367 | [0.005720, 0.007015] | 0.909 | 100% | supported | met |
| Ridge | Block | Block | 0.000875 | [0.000776, 0.000973] | 0.806 | 75% | supported | not met |
| Ridge | Degree | Degree | 0.009685 | [0.008970, 0.010400] | 0.957 | 100% | supported | met |
| MLP | Edge | Edge | 0.006328 | [0.005640, 0.007016] | 0.906 | 100% | supported | met |
| MLP | Block | Block | 0.000805 | [0.000682, 0.000927] | 0.725 | 65% | supported | not met |
| MLP | Degree | Degree | 0.009613 | [0.008863, 0.010363] | 0.961 | 100% | supported | met |
| GCN | Edge | Edge | 0.014044 | [0.013091, 0.014996] | 0.949 | 100% | supported | met |
| GCN | Block | Block | 0.000848 | [0.000679, 0.001016] | 0.790 | 65% | supported | not met |
| GCN | Degree | Degree | 0.011087 | [0.010247, 0.011927] | 0.972 | 100% | supported | met |

All nine Holm-adjusted directional p-values are `4.01e-05` within the pre-specified
21-test family.

## Null negative control

The ridge null has no detected ladder step. MLP and GCN detect a negative density step:
adding an irrelevant ER graph is worse than the matched no-graph model. The density step
is negative in all 20 null seeds for both neural families. This is not a false attribution
of graph benefit; it is evidence that irrelevant graph channels can impose an optimisation
or variance cost. No later null step (degree, block, or edge) is detected for either neural
family.

## Interpretation

The experiment supports the weaker and defensible construct-validity claim:

> Across matched ridge, MLP, and GCN experts, the expected structural level is the dominant
> mean attribution and has a positive Holm-supported effect in every controlled signal
> scenario.

It does not support a deterministic or seed-wise claim that PCGA-L isolates at least 70% of
the positive attribution in every model/scenario cell. Block signal is weaker and its
finite-sample attribution sometimes spreads into adjacent ladder steps. The appropriate
paper claim is therefore **directional hierarchical attribution with uncertainty**, not
perfect source recovery or universal identification.

The negative null density effects also confirm why PCGA-L steps must be allowed to be
negative. A graph level may reduce rather than add predictive value. This is a diagnostic
output, not a failure that should be clipped or hidden.

## Files

- Generated report: `outputs/pcga_l_multimodel_v1/REPORT.md`
- Per-seed effects: `outputs/pcga_l_multimodel_v1/per_seed.csv`
- Confirmatory tests: `outputs/pcga_l_multimodel_v1/hypotheses.csv`
- Recovery concentrations: `outputs/pcga_l_multimodel_v1/recovery_summary.csv`
- Training diagnostics: `outputs/pcga_l_multimodel_v1/seed_*_diagnostics.csv`

