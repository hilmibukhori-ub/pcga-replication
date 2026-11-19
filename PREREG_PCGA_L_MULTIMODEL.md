# Prospective protocol: PCGA-L cross-model construct recovery

**Written:** 16 November 2026, before any confirmatory seed (100--119) was run.

## Status and purpose

This is a prospective construct-recovery experiment for the hierarchical extension of
Placebo-Calibrated Graph Attribution (PCGA-L). It does not test whether real stock-market
relations are useful. It asks whether the ladder attributes predictive performance to the
known generating level across three matched model families.

Development seeds 9000 and 9001 may be used for runtime checks, debugging, and numerical
stability. They are permanently excluded from the confirmatory analysis. After development,
the protocol, configuration, runner, model module, core PCGA-L module, and tests will be
SHA-256 frozen. Confirmatory execution must refuse to run if those hashes change.

## Fixed data-generating process

The four controlled scenarios and coefficients are inherited unchanged from the earlier
placebo-ladder experiment:

- `edge`: the target depends on the exact true neighbours; expected step `edge`;
- `block`: the target depends on the sector mean; expected step `block`;
- `degree`: the target depends on node degree times a common factor; expected step `degree`;
- `null`: no relational term; no positive step is expected.

There are 60 assets in four sectors, 2,000 decision times, chronological train/validation/test
splits, and a purged final decision time in every split. Confirmatory DGP seeds are 100--119.

## Fixed graph ladder

For every DGP seed, the same graphs are used by every scenario and model family:

1. L0: no graph; graph inputs/operators are exactly zero;
2. L1: three fixed Erdős--Rényi placebos preserving edge count;
3. L2: three degree-preserving edge-swap placebos;
4. L3: three placebos preserving every node's degree and sector-neighbour counts;
5. L4: the true graph.

All variants are separately trained. Level loss is the mean over its placebo variants. For
losses, the four step effects are L0-L1, L1-L2, L2-L3, and L3-L4. They may be negative and
must telescope exactly to D1 = L0-L4.

## Fixed matched model families

- `ridge`: pooled linear expert with own, neighbour-mean, and degree-scaled neighbour-sum
  channels; alpha is selected from the fixed grid on validation data.
- `mlp`: pooled one-hidden-layer MLP over the same six channels.
- `gcn`: two-layer message-passing network with separate self, neighbour-mean, and
  degree-scaled neighbour-sum pathways.

L0 retains the same six input columns for ridge/MLP and the same GCN parameters as L1--L4;
only graph channels/operators are zero. Neural initialisation and minibatch order are paired
across every graph level and placebo within a family/scenario/DGP seed. Early stopping uses
validation data only. Hyperparameters are fixed in `configs/pcga_l_multimodel_v1.yaml`.

## Confirmatory hypotheses and family

For each of 3 model families × 3 signal scenarios (nine tests), the pre-specified expected
step is positive. A test is supported only when:

1. its Holm-adjusted two-sided Wilcoxon p-value is below 0.05;
2. the mean is positive; and
3. its 95% t interval excludes zero in the positive direction.

The Holm family also includes the four two-sided null-step checks for each model family,
giving 21 tests in total. Null non-rejection is reported as a negative-control check and is
not interpreted as equivalence.

Attribution concentration is a second, pre-specified descriptive criterion. In each of the
nine family/scenario cells, the expected step must account for at least 70% of summed positive
step value in at least 80% of confirmatory seeds. The global recovery criterion is met only if
all nine directional tests and all nine concentration checks pass. Every cell is reported
regardless of direction; failure does not authorise tuning or removal of a model family.

## Interpretation boundary

Passing supports construct recovery for these DGPs and matched experts. It does not prove
causal economic spillovers, universal identification, market usefulness, or materiality.
Failing identifies a model/DGP boundary of the attribution ladder and must be reported.

