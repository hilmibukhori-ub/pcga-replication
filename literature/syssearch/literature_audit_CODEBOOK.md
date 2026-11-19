# Codebook: `literature_audit_hilmi.csv` and `literature_audit_syaiful.csv` (supplementary graph-attribution audit)

**Scope:** the 36 RQ2-sample papers with full text available, one row per paper. The 24 abstract-only
papers are excluded: no conclusion is possible for them.

**Relation to the double-coding protocol:** this sheet is a supplementary audit with sharper questions.
It does **not** replace the frozen κ subset (`rq2_double_coding_subset.csv`,
`../RQ2_DOUBLE_CODING_INSTRUCTIONS.md`).

**Blindness:** code from the full text only. Do not open `rq2_coding.csv` until the sheet is complete, if
any of these papers also appear in your double-coding subset.

## Rules
- Use exactly one category per `qN_*` cell, written as shown below.
- Every answer other than `no` needs a short quote plus its section or page in `qN_*_evidence`.
  Example: `§5.3 Table 4: "w/o graph"`.
- Use `unclear` only when the text is ambiguous. Explain why in the evidence cell.
- Write your name or initials in `coder`.
- **Independence:** each coder fills in their own sheet. Do not look at the other sheet or `rq2_coding.csv` until both are complete.

## Questions and categories

**q1_graph_attribution:** Does the evaluation isolate the information in the *specific* firm-to-firm links,
rather than the benefit of having a relational architecture?

| Code | Meaning |
|---|---|
| `yes` | The observed graph is compared with a control that keeps the architecture and training but breaks the identity of the links. Examples: permuted or relabelled firms, or a placebo graph. |
| `partial` | Only indirect evidence. Examples: comparing relation types or graph constructions, or adding and removing edge sets, without a link-identity control. |
| `no` | None of the above. |
| `unclear` | The text is ambiguous. |

**q2_arch_matched_baseline:** Is the graph removed while the model, features, training budget and
selection procedure stay matched?

| Code | Meaning |
|---|---|
| `clean` | The proposed model with only the graph or relational module removed, and nothing else changed. |
| `partial` | The graph is removed but other parts change as well. Examples: different features, a different budget, or a separately tuned baseline. Also use this when only part of the graph is removed. |
| `no` | No such variant. Only external baselines are reported. |
| `unclear` | The text is ambiguous. |

**q3_placebo_design:** Are graphs randomised or relabelled as a control, and is the model retrained on
them?

| Code | Meaning |
|---|---|
| `matched_retrained` | The random or placebo graph keeps the relevant structure, such as density or degree, and the model is fully retrained on it. |
| `unmatched_or_not_retrained` | A random graph is used, but it does not keep the structure, or it is swapped in at test time only. Random *initialization* of a learned graph also goes here. |
| `no` | No random or placebo graph. |
| `unclear` | The text is ambiguous. |

**q4_statistical_inference:** Are graph-specific effects tested, across seeds or against a permutation
or reference distribution?

| Code | Meaning |
|---|---|
| `graph_specific` | A test of the graph against a no-graph or placebo control. |
| `model_vs_baseline_only` | Tests are reported, but only of the full model against other models. |
| `none` | No significance test. |
| `unclear` | The text is ambiguous. |

**q5_validation_protocol:** Is the choice of epoch and hyperparameters kept separate from the final test
period?

| Code | Meaning |
|---|---|
| `yes` | An explicit validation split, or walk-forward validation, is used for selection. |
| `test_set` | Selection is done on the test period. This can be stated in the paper or visible in the released code. |
| `not_reported` | The paper does not say how selection was done. |
| `unclear` | The text is ambiguous. |

**q6_reproducibility:** Can the baseline and the graph-specific controls be reproduced independently from
released material?

| Code | Meaning |
|---|---|
| `full` | Code, data or graph, *and* the ablation or control scripts are released. |
| `code_only` | Model code is released, but the data, graph or controls are missing. |
| `none` | Nothing is released. |
| `unclear` | A link is given but is dead or incomplete. Note which in the evidence cell. |
