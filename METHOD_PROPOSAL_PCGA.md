# Proposed method: Placebo-Calibrated Graph Attribution (PCGA)

## Recommendation

Use **Placebo-Calibrated Graph Attribution (PCGA)** as the proposed method. PCGA is a
model-agnostic validation method for deciding whether the performance of a relational
forecasting model is attributable to:

1. the availability of cross-node message passing;
2. the specific observed relations; or
3. neither of them.

This is a stronger and safer methodological contribution than a new graph selector. Its
validity does not require the tested graph to improve prediction. A null result is an
informative PCGA diagnosis, whereas it would be a failure for a proposed predictive model.

## Problem PCGA solves

A conventional ablation compares a relational model with a baseline that often changes the
architecture, parameter count, training budget, or tuning procedure. Even a clean no-graph
ablation only establishes whether *some* message-passing structure helps. It does not establish
that the observed company-to-company links contain useful information.

PCGA separates these claims with two matched interventions and requires the full model to be
retrained after each intervention.

## Formal specification

Let `M(G; s)` be the complete training-and-evaluation pipeline run with graph `G` and training
seed `s`. Let `q(M)` be a metric oriented so that larger values are better. PCGA uses:

- `G_real`: the observed relation graph;
- `G_empty`: self-loops only, with cross-node messages removed;
- `G_pi[k]`: constrained placebo graphs drawn from a pre-specified permutation orbit.

The placebo transformation must preserve the graph properties that are *not* being tested. In
the stock application, permutation within industry preserves relation types, directions, degree
structure, and endpoint industries while destroying which company occupies each graph position.

### Axis 1: graph utility

For paired seeds,

`U = E_s[q(M(G_real; s)) - q(M(G_empty; s))]`.

`U` asks whether the real graph improves the same architecture relative to no cross-company
message passing. Inference uses a pre-specified paired test and reports an effect interval.

### Axis 2: relation specificity

For `K` independently generated placebos,

`S = q(M(G_real)) - mean_k q(M(G_pi[k]))`.

The one-sided exact Monte Carlo rank p-value is

`p_placebo = (1 + sum_k I[q(M(G_pi[k])) >= q(M(G_real))]) / (K + 1)`.

Under the pre-specified exchangeability null, this p-value is finite-sample valid. Every graph
variant is retrained; swapping graphs into a model trained on `G_real` is not PCGA.

### Axis 3: decision relevance

Before opening the test output, specify a smallest worthwhile gain `delta_min`, based on data,
maintenance, and deployment costs. PCGA recommends adopting the relation data only when:

- graph utility is supported;
- its effect exceeds `delta_min`; and
- the real graph is in the pre-specified upper tail of its placebo distribution.

This third gate prevents a statistically detectable but economically negligible relation effect
from being presented as deployment value.

## Diagnostic outputs

| Graph utility | Relation specificity | PCGA diagnosis | Decision |
|---|---|---|---|
| supported and worthwhile | supported | relation-specific value | relation claim supported |
| supported | not supported | generic structural value | use a cheaper graph/control structure |
| not supported | supported | specific but non-deployable | relation assignment matters, but graph cost is not justified |
| not supported | not supported | unattributed/no graph value | do not credit performance to relations |

An inconclusive test remains explicitly inconclusive; PCGA does not convert non-rejection into
equivalence or evidence of no effect.

## Evidence already available in this repository

### Construct validity

The corrected synthetic validity experiment distinguishes regimes in the intended direction:

- informative graph: `U = +0.270`, 95% CI `[+0.217, +0.324]`;
- no-signal graph: `U = -0.114`, 95% CI `[-0.168, -0.060]`;
- real-vs-placebo specificity in the informative regime: `+0.266`,
  95% CI `[+0.219, +0.312]`;
- graph corruption lowers utility by `-0.250`, 95% CI `[-0.315, -0.184]`.

These are the supported D-side results in
`outputs/da_synthetic_pilot_validity_v2/`. The failed A2a and PC-SGL selector results are not
part of PCGA and should not be used to motivate it.

### Empirical demonstration

The existing experiments already cover all important PCGA behaviours:

- RSR and RT-GCN show no confirmed graph utility and no relation specificity;
- the RSR placebo ladder produces one statistically supported relation-specific effect that is
  economically negligible and model-sensitive;
- the audit finds no clean placebo-graph control among the 36 independently coded papers.

Thus, the empirical null results demonstrate why the method is needed; they do not invalidate it.

## Novelty position

The defensible novelty claim is the **combination**, not any one ingredient:

> To our knowledge, PCGA is the first pre-registered application of a matched no-graph control
> and retrained, structure-preserving placebo graphs with an exact randomisation test to attribute
> published relational stock-prediction gains.

Do not claim that PCGA is the first use of random graphs, graph ablations, permutation tests, or
placebos. Prior work contains each of those ideas separately. The paper's systematic search and
the two-coder audit support the narrower combination claim.

A targeted literature check on 16 November 2026 found adjacent work that randomises graph
structure or advocates graph-agnostic baselines, but not the complete PCGA design for published
stock-GNN relation claims:

- *Graph Neural Networks Use Graphs When They Shouldn't* studies empty/non-informative graphs:
  https://proceedings.mlr.press/v235/bechler-speicher24a.html
- *Revisiting Graph Learning Benchmarks: When is the Graph Actually Necessary?* compares tuned
  MLPs with graph models and proposes synthetic benchmarks, but does not test specific observed
  relations against constrained placebos:
  https://link.springer.com/article/10.1007/s11063-026-11833-6
- a graph-pooling replication directly randomises input edges to isolate structural information:
  https://openreview.net/pdf?id=WpYdiLd5Fm
- *Towards Understanding and Reducing Graph Structural Noise for GNNs* studies graph noise:
  https://proceedings.mlr.press/v202/dong23a.html
- *Node Feature Kernels Increase Graph Convolutional Network Robustness* compares perturbed
  graphs with MLPs:
  https://proceedings.mlr.press/v151/el-amine-seddik22a.html
- the exact-random-permutation literature supports the identity-plus-randomisations rank test,
  but is a statistical foundation rather than a stock-GNN attribution method:
  https://link.springer.com/article/10.1007/s11749-017-0571-1

This web check is supporting evidence, not a replacement for the paper's frozen systematic search.

## Claims that protect the proposal

Use these claims:

- PCGA is an attribution and validation method, not a new forecasting architecture.
- It operationally attributes predictive performance within a fixed model, dataset, period, and
  placebo orbit.
- The exactness claim applies to the placebo rank test under exchangeability.
- Negative and inconclusive diagnoses are valid method outputs.
- The method is model-agnostic and can be applied before deployment or during monitoring.

Avoid these claims:

- PCGA establishes causal economic spillovers.
- PCGA proves that company relations are universally useless.
- A non-significant result proves equivalence.
- The chosen placebo removes every possible confounder.
- PCGA is the first method ever to randomise or perturb a graph.

## Reviewer stress test

| Likely objection | Protective response |
|---|---|
| “This is only a no-graph ablation.” | PCGA requires a second, constrained placebo axis that tests the observed node-to-node assignment. |
| “Random graphs are not new.” | Novelty is claimed only for the complete matched-control, retrained-placebo, exact-test, pre-registered application to published stock-GNN claims. |
| “The empirical results are null, so the method failed.” | PCGA is a falsification/attribution method; `unattributed` and `inconclusive` are defined outputs. Synthetic informative and no-signal regimes establish directional construct validity. |
| “The result is causal.” | The paper claims operational performance attribution conditional on the model and placebo orbit, not causal spillovers between firms. |
| “The placebo is arbitrary.” | The preserved properties and the destroyed assignment are stated explicitly; alternative placebo orbits answer different, narrower questions. |
| “The supported effect is tiny.” | A pre-specified materiality gate separates statistical specificity from deployment value. |
| “PCGA was named after results were seen.” | The name and unified packaging are post-study; the re-test hypotheses, graph interventions, seeds, and tests were pre-registered and hash-frozen. Do not imply that the acronym itself was pre-registered. |
| “The synthetic study is post hoc.” | Present it only as separate construct validation of the measurement axes, never as confirmatory evidence for the empirical stock claims. |

## Paper integration plan

1. Introduce the name PCGA in the abstract and contributions without changing the empirical
   conclusions.
2. Present the two estimands, exact rank test, and three-gate decision rule as one method.
3. Add a compact PCGA diagnostic table rather than a new predictive benchmark.
4. Use the corrected synthetic D-side results only as method-validity evidence; exclude all
   unsuccessful selector experiments.
5. Keep the novelty claim scoped to published relational stock-prediction claims and the complete
   matched/placebo/pre-registered combination.

## Go/no-go verdict

**GO for PCGA as the paper's proposed method.** It is supported by the current experimental
design, synthetic construct validation, applied re-tests, and literature audit. It does not need
Hyperion or a claim of improved stock prediction. The main manuscript should not be renamed or
rewritten around PCGA until the terminology and compact method presentation are approved by all
authors.
