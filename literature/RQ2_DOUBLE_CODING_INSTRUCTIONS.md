# RQ2: independent second coding (instructions for coder 2)

**Purpose:** check how reliable the single-coder RQ2 audit is (paper §6, Limitations).

**Subset:** 12 papers in `literature/syssearch/rq2_double_coding_subset.csv`. They were drawn uniformly
from the 36 papers that have full text, with `default_rng(20261111)`, before any second coding.

## Rules
1. **Code blind.** Do not open `rq2_coding.csv` (coder 1) until you have finished all 12 papers.
2. **Code from the full text** (PDF or publisher HTML). Fill in the `coder2_*` columns only.
3. **Use exactly these categories:**
   - `no_relation_baseline`: yes / no. Is any baseline without cross-stock relations reported?
   - `same_arch_no_graph`: clean / partial / no.
     - clean: the graph is removed from the proposed model and nothing else changes.
     - partial: the graph is only partly removed, or other parts change as well.
   - `random_or_placebo_graph`: yes / partial / no. Is a random, shuffled, permuted or placebo graph used
     as a control?
   - `n_runs`: 1 / >1 / not stated.
   - `sig_test`: yes / no. Is any significance test reported, against anything?
   - `validation_split_for_selection`: yes / not reported / test set. How were the model and epoch selected?
   - `code_released`: yes / no.
   - `evidence_quotes`: a short quote with its section or page for each non-trivial decision.
4. **When finished,** send the file back. Agreement is then computed per item: raw agreement and Cohen's
   κ, reported in §6 and the Limitations. Disagreements are resolved by discussion and logged. The
   original codes are kept.

## Not allowed
- Changing coder 1's sheet.
- Re-sampling the subset.
