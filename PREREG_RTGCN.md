# Pre-registration — re-test of RT-GCN (Zheng et al., ICDE 2023), Uniform variant, NASDAQ (v0.1)

**Status:** FROZEN 2026-11-10 (see `PREREG_RTGCN_FREEZE.md`). Confirm modes of
`scripts/run_rtgcn_replica.py` refuse to run unless its hashes match.
**Drafted:** 2026-11-10. Spec: `configs/rtgcn_replica.yaml`.
**Code:** `dapilot/rtgcn_replica.py` (harness), `dapilot/rtgcn_report.py` (analysis),
`scripts/run_rtgcn_replica.py` (runner), `tests/test_rtgcn_replica.py`.
**Environment:** `~/.venvs/rsr_arm` (arm64, Python 3.9.6, torch 2.8.0), CPU, one thread per process.
Companion study: `PREREG_RSR_REPLICATION.md` (RSR, Feng et al. 2019).

## 0. Disclosure of what has been seen

- **The RT-GCN paper** (`literature/zheng2023_rtgcn_icde.pdf`, local licensed copy, not for
  redistribution) and **the public code** (github.com/zhengzetao/RTGCN, commit `617c3f53`) have
  been read.
- **The released test arrays (2020)** were opened only for:
  - format conversion (label pickle to npy, through a restricted unpickler that allows numpy
    reconstruct only)
  - shape checks: 172 × 854
  - the min/max of the normalised test *prices*
  - an equivalence unit test that compares two model builds on 8 test inputs (outputs only, no
    metric)
- **No loss, return, IRR, MRR or other metric on the 2020 test period has been computed.**
- **Dev runs** were made inside the training period. The pseudo-test is the last 252 training
  samples (≈ 2019-03 to 2020-02), and the released test arrays are never loaded in dev. Their
  results are reported in §9.
- **Our RSR results are known.** H1 and H2 there were not detected.

## 1. Claim under test

Table IV of the paper, NASDAQ, mean of 15 runs:

| | MRR | IRR-1 | IRR-5 | IRR-10 |
|---|---|---|---|---|
| Rank_LSTM (no relations) | 0.041 | 0.35 | 0.42 | 0.82 |
| **RT-GCN (U)** | **0.055** | **0.45** | **0.71** | **0.82** |
| RT-GCN (T) (best variant, not tested here) | 0.061 | 1.25 | 0.97 | 1.03 |

The paper attributes these gains to modelling relations ("to fully utilize the relations among
stocks"). RT-GCN (U) is the variant in the released config. Variant (T) needs about 15 GB per
batch on CPU, which does not fit in this machine's 16 GB, so it is out of scope (§12).

**Facts about the published protocol.** These are stated neutrally, as observations about the
paper and the public code, and are verifiable:
1. The paper describes only a train/test split: "chronologically split ... into two time periods
   for training and testing". Hyperparameters were chosen "by grid search" with no validation
   period mentioned.
2. The code (`processor/recognition.py`) keeps the epoch with the lowest **test** loss and reports
   its metrics.
3. The code deletes test day 89 (`np.delete(..., 89)`) before computing metrics.
4. The released `train.yaml` uses Adam with lr 0.1 and weight decay 0.1. The paper states Adam with
   lr 0.001.
5. The code sets no random seed.
6. The released test data has 172 samples. Table II of the paper lists 207 test days.

## 2. Data, code and provenance

- Code: commit `617c3f53a58753251929a0cdbd1c12f7e3df8154`. Data: the authors' Google Drive folder,
  NASDAQ subfolder, downloaded on 2026-11-10 to `data/raw/rtgcn_dl/NASDAQ`. sha256 values are in
  the freeze record.
- Train: 1295 samples (2015-01 – 2020-02-28). Test: 172 samples (2020). 854 stocks. Each sample
  is a 15-day window with 5 features, of which channels [0, 1, 2, 4] are used, as in the code.
- Relation tensor: 854 × 854 × 137.
  - Channels 0–39 are Wiki relations.
  - Channels 40–135 are industry. These are cliques, disjoint, with membership on the diagonal.
    Verified: industry edges are exactly the same-label pairs.
  - Channel 136 is identity.
- Industry labels are derived from the diagonal of channels 40–135. That gives 96 industries, with
  15 stocks belonging to none.

## 3. Model and training

- **Model:** the original `net.rt_gcn.Model`, imported unchanged. The harness patches only
  `Tensor.cuda`, making it a no-op, and replaces `Graph.get_edge` so that it accepts a
  precomputed edge matrix. The test `test_patched_model_equals_original` shows that the
  parameters, A and outputs are identical to the original build.
- **Loss:** the same as `net/loss.py` (MSE + 0.1 · pairwise rank loss), tested.
- **Training:** follows `recognition.py`.
  - batch 64, shuffle, drop_last; 100 epochs
  - prediction = (out − price)/price
  - evaluation every epoch
  - test loss is the mean of per-batch losses
- **Configs:**
  - **`paper` (PRIMARY):** Adam with lr 1e-3, as in the paper. The paper does not state the
    weight decay, so the code's default of 1e-4 is used. This is an assumption.
  - **`released` (SENSITIVITY):** Adam with lr 0.1 and weight decay 0.1, as in `train.yaml`.
  - The choice of primary was made from the dev rehearsal (§9), where the released config trained
    unstably (validation loss jumped from 0.07 to 2.84).
- **Seeds:** seed = 20261111 + s (torch and numpy), with the same s for every graph (common random
  numbers). The data order comes from a torch generator seeded the same way.

## 4. Arms and splits

| arm | train | validation | test | epoch selection |
|---|---|---|---|---|
| **val** (proper) | samples 0–1041 | samples 1043–1294 (last 252; one sample purged) | released test (172) | min **val** loss, plus test-loss and last-epoch as comparisons |
| **pub** (published protocol) | all 1295 | none | released test | min **test** loss (as in the code) |

Each run stores every epoch's losses and daily metrics, so all selection rules are applied after
training, in `rtgcn_report.py`.

## 5. Graphs

- **real:** eye + any relation (Wiki and industry), exactly as `Graph.get_edge` builds it.
- **none:** A = I, which removes all cross-stock propagation. This is the matched no-relation
  baseline: the same network and training. It is not the paper's Rank_LSTM.
- **ind:** industry relations only (descriptive).
- **perm k, k = 0…18:** within-industry node permutation of the full relation adjacency.
  - The 15 no-industry stocks are pooled into one group.
  - The permutation uses `default_rng([20261111, 7, k])`.
  - Industry cliques are invariant, so only the Wiki relations move. This is tested.
  - Under H0, "the assignment of Wiki relations within industries carries no information", the
    real graph is exchangeable with the placebos, so the rank test is exact.

## 6. Metrics

- **IRR-k:** Σ over the 172 test days of the mean realised return of the top-k predicted stocks,
  as in `show_return_ration` (its `bt_k` = 1 + IRR-k).
  - **All days are used for the hypothesis tests.**
  - The "pub" variant deletes day 89, as the code does, and is used for R0.
- **MRR:** as in `show_MRR`.
- **Primary metric: IRR-1.** It is the paper's headline IRR for this comparison. IRR-5, IRR-10 and
  MRR are secondary.

## 7. Hypotheses

**Confirmatory family. Holm over 2 tests at α = 0.05. Paper config, val arm, 20 seeds.**

| ID | H1 | statistic | test |
|---|---|---|---|
| **H-A** | test-loss epoch selection inflates IRR-1 | IRR-1(e_test) − IRR-1(e_val), real graph, paired by seed | one-sided paired t |
| **H-B** | relations improve IRR-1 (D1) | IRR-1(real, e_val) − IRR-1(none, e_val), paired by seed | one-sided paired t |

**Separate exact test H-C (edge):**
- Statistic: T = IRR-1 at e_val, seed 0, compared between real and the 19 placebos.
- p = (1 + #{k : T(perm_k) ≥ T(real)}) / 20, at α = 0.05. The smallest possible p is 0.05.
- It is reported apart from the Holm family.

**Calibration note.** The seed-paired t-test was A/A-calibrated in the RSR rehearsal, where it
rejected 4.0–4.5% of the time at a nominal 5%. It was not separately A/A-checked for RT-GCN, and
this is disclosed. In the dev run, IRR-1 is almost uncorrelated with loss across epochs
(r ≈ 0.0 to −0.2). The paper therefore expects H-A to be small. It is pre-registered anyway,
because a null result is informative.

## 8. Interpretation rules (fixed now)

1. **H-B not detected:** the relation gain of RT-GCN(U) does not replicate under a matched
   no-relation baseline and validation-based selection. It is always reported with the estimate,
   the 95% CI, and the paper's +0.10 versus Rank_LSTM, noting that Rank_LSTM is a different
   baseline.
2. **H-A not detected while the variance is large:** reported as "loss-based selection, on test or
   on validation, does not systematically inflate IRR. Single-run IRR is dominated by epoch-to-epoch
   and seed-to-seed noise", together with the variance decomposition (§10.4).
3. **H-A supported:** the inflation is reported with its size relative to the paper's claimed
   gains.
4. **H-C:** if supported, the specific Wiki relations matter beyond the industry structure. Wiki
   relations were "collected before the test period" according to the paper, which was not
   verified. If not supported, it is reported with the placebo distribution.
5. **R0:** the pub-arm mean (10 seeds, day 89 deleted) is compared with Table IV. The paper gives no
   sd. A "reproduction gap" is declared if the paper's value lies outside our 95% CI of the mean.
   Nothing is retuned.

## 9. Dev rehearsal (pre-freeze, no 2020 data)

**Learning-rate check** (real graph, seed 0, dev split, 100 epochs, `outputs/rtgcn_dev`):

| | released (lr 0.1, wd 0.1) | paper (lr 1e-3, wd 1e-4) |
|---|---|---|
| train loss, epochs 0 → 99 | 0.69 → 0.032 (stalls from about epoch 10) | 0.37 → 0.0042 |
| val loss | minimum 0.067; **spike to 2.84 at epoch 49** | falls smoothly to 0.013 |
| IRR-1 across epochs (pseudo-test) | 1.08 ± 0.61 (range −0.71 to 2.71) | 1.32 ± 0.66 (range −0.21 to 3.42) |
| corr(test loss, IRR-1) across epochs | −0.19 | 0.00 |
| IRR-1 at val / test-loss / last epoch | 1.75 / 1.28 / 1.33 | 0.97 / 1.69 / 0.24 |

**Pipeline checks:**
- A 4-epoch smoke run covered every code path: dev/devpub, both configs, real/none/ind/perm
  (`outputs/rtgcn_smoke`). The report runs end to end, and the H-A, H-B and H-C values were
  checked by hand against the run logs.
- Determinism: the same seed gives the identical epoch-1 IRR in separate runs.
- Kill/resume: a killed run leaves no partial file (atomic rename), and resuming skips finished
  models.
- The freeze guard refuses confirm mode before the freeze.
- The 4 unit tests pass.

**Pre-freeze fixes:**
- The freeze-hash regex was corrected. It had doubled backslashes and matched nothing, so it
  failed safe.
- Output naming was unified between the runner and the report.
- A `devpub` mode was added so that the pub path could be rehearsed.

**Pre-launch amendment RT1 (2026-11-10, before any confirm run; see DEVIATIONS RT1):** this adds
per-epoch checkpointing (bit-identical resume, tested), per-model error isolation, a resumable RSR
sweep, and a second RT-GCN pass in the launcher. No computation or analysis changed. The new freeze
record supersedes `PREREG_RTGCN_FREEZE_v0.1_superseded.md`.

## 10. Secondary and descriptive analyses

1. IRR-5, IRR-10 and MRR versions of H-A and H-B, and last-epoch selection.
2. **Released-config sensitivity** (val arm, 10 seeds, real and none): D1 and inflation, paired t,
   descriptive.
3. Industry-only D1 (5 seeds), descriptive.
4. Variance decomposition of IRR-1: within-run sd across epochs, between-seed sd at e_val, mean
   corr(test loss, IRR-1), and mean oracle best-epoch IRR-1.
5. R0 (pub arm, 10 seeds): IRR-1/5/10 and MRR against Table IV, with day 89 deleted.

## 11. Execution

- Models: val/paper real ×20, none ×20, perm ×19 (s0), ind ×5; pub/paper real ×10; val/released
  real ×10, none ×10. That is 94 models, at roughly 25–40 min each, run in 6 parallel queues.
- **Launch starts after the RSR stage B finishes**, through a waiting launcher, so the two studies
  do not compete for CPU.
- When done: `~/.venvs/rsr_arm/bin/python scripts/rtgcn_summarise.py`. The launcher is
  `scripts/launch_rtgcn_after_rsr.sh`, running in screen `rt_launcher`. Deviations go in `DEVIATIONS.md` with the prefix RT.

## 12. Known limitations

- NASDAQ and variant U only. NYSE and variant T need about 2.7× more memory and time; they could be
  added on a GPU later as a separate, labelled extension.
- The paper's weight decay is unknown and assumed to be 1e-4.
- The released test data has 172 samples against the paper's 207 test days, and it is used as
  released.
- The matched no-relation baseline is the same network with A = I, not the paper's Rank_LSTM.
- A single market and test period (2020, which includes the COVID crash and recovery).
