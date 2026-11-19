# Codebook for independent verification of the 36-paper RQ2 audit

## General rule

Code what the paper actually reports, not what the method might permit and not what a repository might
contain unless the question explicitly concerns released code. Apply exactly one permitted category per
question. Record evidence with a page, section, table, figure, appendix, or repository location.

## 1. `no_relation_baseline`

**Question:** Is any baseline without cross-stock relations reported?

Permitted values:

- `yes`: at least one reported baseline does not use cross-stock relations.
- `no`: no such baseline is reported.

A different architecture can count here. This question asks only whether any non-relational baseline is
reported; architectural matching is assessed separately.

## 2. `same_arch_no_graph`

**Question:** Is the proposed model evaluated after removing the graph or relational component while
keeping the rest of the architecture and evaluation matched?

Permitted values:

- `clean`: only the graph or relational component is removed; features, remaining architecture,
  training budget, and selection procedure remain matched.
- `partial`: the graph is only partly removed, or removing it also changes other material components.
- `no`: no same-architecture no-graph evaluation is reported.

When the paper does not provide enough information to establish a clean removal, prefer `partial` and
explain what is unclear.

## 3. `random_or_placebo_graph`

**Question:** Is a random, shuffled, permuted, relabelled, or placebo graph used as a control for the
observed relations?

Permitted values:

- `yes`: a genuine randomised/relabelled/placebo relation graph is used as a control and the comparison
  bears on the identity of the observed links.
- `partial`: a related but incomplete control is reported, such as random initialisation of a learned
  graph, an unmatched random graph, or a graph changed together with other components.
- `no`: no such control is reported.

Do not count ordinary random weight initialisation, dropout, random train/test splits, or a learned graph
initialised randomly as `yes`.

## 4. `n_runs`

**Question:** How many independent training runs or random seeds are reported for the relevant model
results?

Permitted values:

- `1`: one run is reported or the presentation clearly represents a single run.
- `>1`: more than one independent run/seed is explicitly reported.
- `not stated`: the number of runs or seeds cannot be determined from the paper.

Cross-validation folds, time windows, assets, and repeated test dates do not by themselves count as
independent training runs.

## 5. `sig_test`

**Question:** Is any formal statistical significance test reported for predictive/model comparisons?

Permitted values:

- `yes`: a named test, p-value, confidence procedure, or equivalent formal inferential test is reported.
- `no`: no formal significance test is reported.

Error bars, standard deviations, multiple seeds, or boldface best results alone do not count as a
significance test. The test may compare the proposed model with any baseline; whether it is graph-specific
should be described in the evidence field.

## 6. `validation_split_for_selection`

**Question:** How are hyperparameters, model variants, or training epochs selected relative to the final
test set?

Permitted values:

- `yes`: an explicit validation split or walk-forward validation is used for selection.
- `not reported`: the selection data/procedure is not stated clearly enough to determine.
- `test set`: the final test set is explicitly used to select hyperparameters, variants, checkpoints, or
  the reported epoch.

If both validation and test-set selection occur, use `test set` and explain the sequence.

## 7. `code_released`

**Question:** Is implementation code for the reported method publicly released or supplied with the
paper?

Permitted values:

- `yes`: an identifiable code repository or supplementary code is provided and attributable to the
  paper/method.
- `no`: no implementation code is provided, or the paper only links to data or to another method's code.

If a stated repository link is dead or inaccessible, use `no` and record the link and access problem in
the evidence field.

## Evidence conventions

- Preserve quotations exactly but keep them short.
- Prefix each item with a location, for example `p. 8, Table 3:` or `§4.2:`.
- For `no`, `none`, or `not stated`, record the places checked, for example `Methods §3, Experiments §4,
  Tables 2--5 checked; no run count stated`.
- Separate multiple evidence items with ` | ` inside a CSV cell.
- If the PDF page number differs from the printed page number, report both when practical.

