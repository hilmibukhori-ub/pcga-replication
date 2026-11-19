# Pre-registration addendum — RT-GCN re-test EXTENSION: NYSE and the Time-aware variant (v0.1)

**Status:** FROZEN 2026-11-10 (see `PREREG_RTGCN_EXT_FREEZE.md`). The confirm modes of
`scripts/run_rtgcn_ext.py` refuse to run unless its hashes match.
**Parent:** `PREREG_RTGCN.md` (main study: RT-GCN(U), NASDAQ), frozen 2026-11-10. Everything that is not
restated here is inherited unchanged: loss, training loop, splits, metrics, epoch-selection rules, the
interpretation rules of §8, and the placebo construction.
**Spec:** `configs/rtgcn_ext.yaml`.
**Code:**
- `dapilot/rtgcn_ext.py`: harness. It imports the frozen `dapilot/rtgcn_replica.py` and does not modify it.
- `dapilot/rtgcn_ext_report.py`: analysis.
- `scripts/run_rtgcn_ext.py`: runner.
- `tests/test_rtgcn_ext.py`.
- `scripts/hyperion/rtgcn_ext.sbatch`: execution.

**Label in the paper:** *pre-registered extension*. Its results are reported in their own section. They
can broaden or qualify the main study's conclusion, but they do not change the main study's verdicts.

## 0. Disclosure of what has been seen (at freeze time)

- Everything disclosed in `PREREG_RTGCN.md` §0.
- **NYSE released data.** This was downloaded on 2026-11-10 directly to the HPC system. The label pickles
  were converted with `scripts/rtgcn_convert_labels.py`, which runs an opcode allow-list scan and then a
  restricted unpickler. Shapes, dtypes and channel layout were checked (§2). Summary statistics of the
  labels were printed by the converter: train mean +3.1e-07 and sd 2.2e-02; **test mean +2.3e-03 and sd
  7.4e-02**. The test sd was printed by the converter and is disclosed here. No model was evaluated on
  the 2020 NYSE data.
- **Dev smoke runs** (§9). These were 2 epochs on the dev split, inside the training period, for timing,
  memory and code-path checks. The runner printed per-epoch dev pseudo-test top-1 sums. These are
  training-period numbers and are disclosed.
- **Main-study (NASDAQ U) results.** The main confirm runs were executing on the HPC system while this
  addendum was written. **The design in this document was fixed before any main-study result was
  read.** The analysis author did not open the run logs or the npz files, and the progress checks read
  only file counts and checkpoint epoch indices. The design-freeze time is recorded in the freeze file.

## 1. Claims under test

Table IV of the paper (mean of 15 runs):

| cell | MRR | IRR-1 | IRR-5 | IRR-10 | Rank_LSTM IRR-1 |
|---|---|---|---|---|---|
| NYSE, RT-GCN (U) | 0.040 | 0.65 | 0.88 | 0.96 | 0.30 |
| NASDAQ, RT-GCN (T) | 0.061 | **1.25** | 0.97 | 1.03 | 0.35 |
| NYSE, RT-GCN (T) | 0.056 | 0.92 | 1.10 | 1.13 | 0.30 |

RT-GCN (T) is the paper's headline model ("Improvement 40.4%" in IRR-1 on NASDAQ).

**Additional facts about the released code, stated neutrally:**
1. `config/rt_gcn/NYSE/train.yaml` sets `edge_importance_weighting: 'Time-aware'` and `batch_size: 32`,
   but its `relation_path` points to `NASDAQ_relation.npy`. That file has 854 nodes, while the NYSE
   graph has 1405, so this config cannot run as released. We use the NYSE relation file.
2. The Time-aware forward pass uses `nn.MultiheadAttention(..., attn_mask=mask)`, where
   `mask = 1[Σ_d relation > 0]` is a **float** mask.
   - In PyTorch a float `attn_mask` is **added** to the attention logits. So related pairs get +1 and
     unrelated pairs get 0. This is a soft bias, not a hard mask.
   - The attention weights are then multiplied by `Linear(D→1)(relation) + I`.
   - The paper's Eq. (5) instead describes the scaled dot product times (Aᵀw + b).
   - **We test the released code as it is.**

## 2. Data and provenance (NYSE)

- Source: the authors' Google Drive folder, `data/NYSE`, downloaded on 2026-11-10. The sha256 values are
  in the freeze record.
- Arrays:
  - train EOD (1295, 5, 15, 1405); test EOD (172, 5, 15, 1405)
  - prices and labels (·, 1405)
  - `NYSE_relation.npy` (1405, 1405, 135)
- Verified relation layout:
  - The last channel is the identity.
  - The non-self channels equal the concatenation of `NYSE_wiki_relation[..., :-1]` (27 channels,
    indices 0–26) and `NYSE_industry_relation[..., :-1]` (107 channels, indices 27–133).
  - The industry channels are disjoint cliques with membership on the diagonal. Their edges are exactly
    the same-label pairs. There are 107 industries, and 21 stocks have no industry. Labels are stored in
    `industry_labels_derived.npy`.
  - Wiki: 6,827 directed off-diagonal pairs.
- The structure is the same as NASDAQ (Wiki 0–39, industry 40–135).

## 3. Model and training

- **Variant U:** as in the main study.
- **Variant T:** the original `net.rt_gcn.Model` with `edge_importance_weighting='Time-aware'`, imported
  unchanged.
  - `Graph.get_edge` is replaced so that it installs a given relation tensor (V × V × D, the non-self
    channels, float32) and the edge matrix.
  - `test_patched_equals_original` checks that the parameters and the outputs on real inputs equal those
    of the model built by the original Graph from the released file.
- **Config:** the main study's primary `paper` config (Adam, lr 1e-3, wd 1e-4) and 100 epochs.
  **Batch size** follows the released `train.yaml` of each market: **NASDAQ 64, NYSE 32**. No
  released-config (lr 0.1) arm is run.
- **Seeds:** seed = 20261111 + s, the same scheme as the main study (common random numbers across graphs).
- **Threads:** a fixed number per cell (§11), enforced by the runner in confirm modes. Results are
  deterministic for a given thread count, and every arm of a cell uses the same count.
- **Platform:** DIPC hyperion, Linux x86_64, Python 3.10, torch 2.1.2 (module `PyTorch/2.1.2-foss-2022b`).
  This is the same as the main study's confirm runs.

## 4. Arms and splits

These are identical to the main study §4 (val arm: train 0–1041, val 1043–1294, test = released 2020;
pub arm: train all 1295, epoch by test loss). The NYSE arrays have the same sample counts.

## 5. Graphs (per cell)

- **real:** the released relations. For U, the edge is eye + 1[any relation]. For T, the full relation
  tensor (and the same edge, which T does not use).
- **none:** the matched no-relation baseline.
  - U: A = I, as in the main study.
  - T: an **all-zero relation tensor**. The attention bias is then 0 everywhere and the relation weight
    is b + I. The same attention network attends over all stocks but receives no relation information.
    This isolates the contribution of the relation data, which is what the paper claims. It is not the
    paper's Rank_LSTM.
- **perm k:** a within-industry node permutation of the whole relation tensor (all channels),
  `default_rng([20261111, 7, k])`.
  - The k-th permutation is the same as in the main study for NASDAQ.
  - Industry channels are invariant and only Wiki relations move (tested).

## 6. Metrics

As in the main study. **IRR-1** is primary. All 172 test days are used for the tests. The pub variant
(day 89 deleted) is used for R0.

## 7. Hypotheses

**Extension confirmatory family: Holm over 6 tests at α = 0.05.** These are H-A and H-B in each of the 3
cells, defined exactly as in the main study §7: one-sided paired t over seeds, val arm.

| ID | cell | H1 |
|---|---|---|
| H-A_NYSE_U, H-A_NASDAQ_T, H-A_NYSE_T | each | test-loss epoch selection inflates IRR-1 (real graph) |
| H-B_NYSE_U, H-B_NASDAQ_T, H-B_NYSE_T | each | relations improve IRR-1 (real vs none, e_val) |

**H-C (edge), per cell:** an exact placebo-rank test at seed 0 with K placebos, unadjusted, reported
apart from the family, as in the main study.

## 8. Interpretation rules

These are inherited from the main study §8, applied per cell. Additionally:

1. The extension and the main study are **not** pooled into one test.
2. If H-B is not detected in the T cells, "the relation gain of the paper's best variant does not
   replicate under a matched no-relation baseline" is reported per market, with the estimate, the CI,
   and the paper's gain over Rank_LSTM (+0.90 NASDAQ and +0.62 NYSE IRR-1). Rank_LSTM is a different
   baseline, and this is stated.
3. If H-B is supported in a T cell, it is reported as evidence that the Time-aware use of relations
   helps in that market, qualified by H-C. If H-C is not supported, the gain is not specific to the
   actual Wiki assignment.

## 9. Dev rehearsal (pre-freeze, no 2020 data)

These were SLURM job 7944034 on bcam-exclusive (`scripts/hyperion/ext_smoke.sbatch`): dev split, real
graph, seed 0, 2 epochs, in `outputs/rtgcn_ext_smoke`.

- **Tests:** `tests/test_rtgcn_ext.py` and `tests/test_rtgcn_replica.py` give **13 passed** on hyperion.
  These cover:
  - patched = original for NYSE-U, NASDAQ-T and NYSE-T (parameters, and outputs with zero tolerance)
  - the channel layout and that industry cliques equal the label pairs (both markets)
  - that the placebo moves only Wiki, with the edge count preserved
  - that `none` has no relation
  - the 5 main-study tests (on the cluster's torch 2.1.2)
- **Timing and memory** (per epoch on the dev split, whose train set is 790 samples; the val arm has
  1041 and the pub arm 1295):

| cell | 1 thread | 4 threads | peak RSS |
|---|---|---|---|
| NYSE-U (bs 32) | 53 s | 18 s | 5.4 GB |
| NASDAQ-T (bs 64) | 209 s | 73 s | 16 GB |
| NYSE-T (bs 32) | 562 s | not completed before freeze | 22 GB |

  The projected time per val-arm model (100 epochs) is about 2 h for NYSE-U (1 thread), about 2.7 h for
  NASDAQ-T (4 threads), and about 7 h for NYSE-T (4 threads, extrapolated from the 1→4-thread speed-up
  of about 2.85 measured on the other cells).
- **Determinism:** the same thread count gives the same numbers. A different thread count changes the
  results at the 3rd–4th significant digit (dev val loss 0.96 vs 0.958), so threads are fixed per cell.
- **Code paths** for all 3 cells (U and T, NASDAQ and NYSE, real) ran end to end, including
  checkpointing. none and perm reuse the same path with a different (edge, relation) and are covered by
  the tests. The analysis module was run end to end on synthetic npz files.
- **Observation (dev, training period):** the NYSE test-loss scale is large (dev pseudo-test loss about
  5–20), consistent with the heavy-tailed NYSE labels. Nothing was changed because of it.

## 10. Secondary and descriptive

These are per cell. IRR-5, IRR-10 and MRR versions of H-A and H-B, and last-epoch selection. A variance
decomposition. R0: the pub arm against Table IV, with a "reproduction gap" if the paper's value lies
outside our 95% CI of the mean.

## 11. Execution

The counts were chosen from compute cost alone (§9), before any extension or main-study result was seen.

| cell | seeds (real, none; val) | placebos (seed 0) | pub seeds | threads | models |
|---|---|---|---|---|---|
| NYSE_U | 20 | 19 | 10 | 1 | 69 |
| NASDAQ_T | 10 | 19 | 5 | 4 | 44 |
| NYSE_T | 10 | 19 | 5 | 4 | 44 |

- In total this is 157 models, roughly 1,900 core-hours.
- The T cells get 10 seeds rather than 20 because of their cost. The paired t is valid with 10 pairs.
  Its lower power is a stated limitation. A null result in a T cell is read together with its CI, and
  never as proof of no effect.
- Each model is one SLURM array task on bcam-exclusive (`scripts/hyperion/rtgcn_ext.sbatch`, task list
  `scripts/hyperion/rtgcn_ext_tasks.txt`).
- Cells are submitted in the order NYSE_T, NASDAQ_T, NYSE_U, longest first.
- Re-submitting the same range fills gaps (skip and resume).
- **Completion rule:** a cell's tests are computed only when all of its pre-registered models exist. If
  a model cannot finish (for example because of node failures), it is re-run with the same seed. If
  that is impossible, the cell is reported as incomplete, and no test is computed on a subset.
- Analysis: `dapilot.rtgcn_ext_report.summarise(cfg)`, with `cfg = configs/rtgcn_ext.yaml`.
  Deviations go in `DEVIATIONS.md` with the prefix RX.

## 12. Known limitations

- The batch size differs between markets, following the released configs.
- The T forward pass follows the released code, which deviates from the paper's Eq. (5) (§1, fact 2).
- RT-GCN (W) is not tested.
- The T cells have 10 seeds (lower power than the main study's 20), and the T and NYSE pub arms have 5
  seeds.
- K = 19 placebos gives a minimum p of 0.05 for H-C.
- The HPC platform is the same as the main study's confirm runs (DEVIATIONS RT2).
