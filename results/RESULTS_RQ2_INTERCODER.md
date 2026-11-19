# RQ2 intercoder reliability (pre-reconciliation)

Date computed: 16 November 2026.

## Inputs

- Coder 1 (Hilmi): `literature/syssearch/rq2_coding.csv`
- Coder 2 (Syaiful): `literature/03_CODING_SHEET_syaiful_CLEAN.csv`
- Frozen codebook: `literature/rq2_independent_verification_syaiful_FROZEN36/01_CODEBOOK.md`
- Reproducible calculation: `python3 scripts/rq2_intercoder_reliability.py`

The sheets were joined by DOI. Syaiful independently coded all 36 full-text papers without access to
Hilmi's codes or the manuscript's audit results. No original coding entry was changed.

## Normalisation of the legacy first-coder sheet

Hilmi's original sheet predates the frozen categorical template and contains descriptive text. The
calculation maps it deterministically to the frozen categories. In particular, a legacy `yes` for the
same-architecture item maps to `clean`; the two legacy `partial` validation entries map to `not
reported` because they do not establish that validation data were used for selection; `correlation test
only` maps to no significance test for a predictive/model comparison; and data-only or promise-only
release statements map to no implementation-code release. These rules are implemented and commented in
the calculation script.

## Pre-reconciliation agreement

| Item | Agreement | Cohen's kappa | Disagreements |
|---|---:|---:|---:|
| No-relation baseline | 34/36 (94.4%) | 0.721 | 2 |
| Same-architecture no-graph | 14/36 (38.9%) | 0.064 | 22 |
| Random or placebo graph | 36/36 (100.0%) | 1.000 | 0 |
| Number of runs | 33/36 (91.7%) | 0.828 | 3 |
| Significance test | 35/36 (97.2%) | 0.929 | 1 |
| Validation selection | 24/36 (66.7%) | 0.435 | 12 |
| Code released | 33/36 (91.7%) | 0.786 | 3 |

Overall raw agreement is **209/252 (82.9%)**. There are **43 disputed cells** before reconciliation.
Agreement is weakest for whether a no-graph comparison is clean and, to a lesser extent, for how model
selection used validation or test data. The central placebo-control classification has perfect
agreement: both coders identify one partial case and no clean random/shuffled/placebo-graph control.

## Same-architecture item: pattern of disagreement

Computed on 19 November 2026 from the same two sheets and the same normalisation
(`scripts/rq2_intercoder_reliability.py`: `normalise_coder1`). Rows: coder 1; columns: coder 2.

| Coder 1 \ Coder 2 | clean | partial | no | Total |
|---|---:|---:|---:|---:|
| clean | 3 | 10 | 1 | 14 |
| partial | 1 | 8 | 5 | 14 |
| no | 1 | 4 | 3 | 8 |
| Total | 5 | 22 | 9 | 36 |

Coder 1 codes 14 ablations as clean and coder 2 codes 5. Of the 22 disagreements, 20 are between adjacent
categories (11 clean/partial, 9 partial/no) and 2 are clean/no. Clean same-architecture ablations are a
minority under either coder (14/36 or 5/36) and after adjudication (4/36).

## Adjudication status

These remain the pre-reconciliation reliability statistics; neither original coding sheet was
overwritten. All 43 differences were subsequently adjudicated against the quoted full-text evidence
using the frozen codebook. The resolution log is `literature/RQ2_JUSTIFICATION.xlsx`.

The adjudicated 36-paper counts used in the manuscript are:

| Item | Adjudicated result |
|---|---:|
| No-relation baseline | yes 33; no 3 |
| Same-architecture no-graph | clean 4; partial 26; no 6 |
| Random or placebo graph | partial 1; no 35 |
| Number of runs | >1 23; not stated 13 |
| Significance test | yes 10; no 26 |
| Validation selection | validation 14; test set 7; not reported 15 |
| Implementation code released | yes 10; no 26 |
