# Pre-registration addendum — Revision 1 robustness: size-stratified placebos (RSR) and K = 99 (RT-GCN U) (v0.1)

**Status:** FROZEN 2026-11-16 (see `PREREG_REV1_FREEZE.md`). The confirm mode of the new runner and every
study-K SLURM task refuse to run unless its hashes match.
**Parents:** `PREREG_RSR_REPLICATION.md`, `PREREG_RTGCN.md`, `PREREG_RTGCN_EXT.md` (all frozen). Everything
that is not restated here is inherited unchanged: models, training, splits, seeds, metrics, epoch selection,
the placebo-rank statistic and the interpretation rules.
**Spec:** `configs/rev1.yaml`.
**Code (to be written before the freeze):**
- `dapilot/rev1_sizebin.py`: the size measure and the industry × size-bin permutation groups.
- `scripts/run_rev1_rsr_sizebin.py`: runner for study S. It imports the frozen `dapilot/rsr_replica.py`
  and does not modify it.
- `dapilot/rev1_report.py`: analysis for both studies.
- `tests/test_rev1.py`.
- `scripts/hyperion/rev1_k99.sbatch`, `scripts/hyperion/rev1_k99_tasks.txt`: execution of study K. Study K
  calls the **frozen runners unchanged**.

**Label in the paper:** *post hoc robustness analysis, pre-registered before it was run (Revision 1)*. It
is reported in its own subsection. It does not change any verdict of the parent studies.

## 0. Why this addendum exists, and what has been seen

**Trigger.** A reviewer of the ESWA submission raised two points:
1. The exactness of the placebo-rank test is conditional on the permutation null. Firm size and liquidity
   may be associated with a stock's position in the graph, and the within-industry permutation breaks that
   association. The reviewer asked for a sensitivity test with placebos that also preserve firm
   characteristics such as size.
2. The RT-GCN edge test has K = 19, so its smallest attainable p is 0.05 and it could not enter a Holm
   family. The reviewer asked for more placebos or more seeds.

**What has been seen (everything).** The analysis author has read all confirmatory and secondary results
of the three parent studies. These include the following.
- RSR edge: the real Wiki graph ranks 17th of 100 (NASDAQ) and 26th of 100 (NYSE).
- RT-GCN H-C: p = 0.45 (NASDAQ U), 0.80 (NYSE U), 0.50 (NASDAQ T) and 0.40 (NYSE T).
- The RT-GCN NYSE day-89 outlier (DEVIATIONS RX-S1).

This addendum is therefore **post hoc in motivation**. It is binding in execution: the design below is
fixed before any model of this addendum is trained.

**New, return-free numbers seen while designing (2026-11-16).** These were computed from raw close × volume
over the RSR training window, without any return or model output:
- Spearman(Wiki degree, log median dollar volume) = 0.39 (NASDAQ) and 0.44 (NYSE).
- Among linked stocks, the same correlation is 0.50 and 0.39.
- The median log dollar volume of linked versus unlinked stocks is 17.8 vs 15.0 (NASDAQ) and 18.3 vs 15.5
  (NYSE).
- With 2 size bins within each industry, 95–98% of linked stocks remain in a group of size > 1.
- With 3 bins, the median group size falls to 2–3.

These numbers motivated the choice of 2 bins (§S.2). They involve no outcome of any model.

## Study S — RSR: size-stratified placebos (reviewer point 1)

### S.1 Size measure

- For each stock i in the RSR universe, a_i is the **median daily dollar volume, Close × Volume**. It is
  computed over the RSR **training window**, which consists of the trading dates
  `{market}_aver_line_dates.csv` rows 29–784 (0-based). These are processed rows 0–755, the rows with index
  below `VALID_INDEX`, from 2013-01-02 to 2015-11-18.
- Source: `data/raw/rsr_repo/data/google_finance/{market}_{ticker}_30Y.csv`. Dates are matched after the
  timezone suffix is stripped, exactly as `preprocess/eod.py` does. Only days with Volume > 0 count.
- A stock with fewer than 50 such days gets a_i = −∞, which places it in the lower bin. At design time no
  stock is affected.
- No validation or test date enters the size measure.
- Dollar volume is a standard proxy for firm size and liquidity. Market capitalisation is not available,
  because the released data contain no shares outstanding. This is stated as a limitation.

### S.2 Placebo construction

- **Groups.** The groups are the parent permutation groups, `rsr_replica.industry_labels` with `n/a`
  pooled. Each group is split into two **size bins**.
  - Stocks are sorted by (a_i, stock index).
  - The lower ⌊n/2⌋ stocks form bin 0, and the rest form bin 1.
  - Stocks alone in a group or bin stay fixed.
- **Placebo k.** π is a uniform random permutation within each (industry, size-bin) cell, drawn with
  `default_rng([20261110, mi, 8, k])`, where mi is 0 for NASDAQ and 1 for NYSE. The tag 8 makes the
  stream distinct from the parent's tag 7. k runs from 0 to 98, so **K = 99**.
- The placebo relation tensor is R^π_ij = R_π(i)π(j), applied with the frozen `rsr_replica.permute_edges`.
- **What it preserves.** It preserves everything the parent placebo preserves. In addition, it preserves
  the size bin of both endpoints of every edge, and hence, approximately, the association between degree
  and size.
- **Exactness.** Under the null that the assignment of Wiki relations within (industry, size-bin) cells
  carries no information about the test metric, the real graph and the K placebos are exchangeable. The
  rank test is then exact. As in the parent study, exactness is **conditional on this null**.

### S.3 Models

- Wiki relations, both markets, seed 0. The model seed is `20261110 + 1000·mi + 0`, the same as the
  parent's real graph at seed 0 (common random numbers).
- Everything else is the frozen RSR_I port: same embedding, same α, 50 epochs, lr 1e-3, and epoch selection
  by validation loss.
- Each market gets 99 placebo models. Each market also gets **one re-run of the real graph at seed 0**, as
  a platform and determinism check (S.6). That is 200 models in total.
- Output directory: `outputs/rev1_rsr_sizebin_v1`.

### S.4 Hypothesis

**H-S_m**, for m in {NASDAQ, NYSE}: the real Wiki graph beats size-stratified placebos.
- T(G) is the test IRR at seed 0 at the validation-selected epoch, exactly as the parent's H3/H4 statistic.
- p = (1 + #{k : T(P_k) ≥ T(real)}) / 100.
- **Family S: Holm over the 2 markets at α = 0.05.**

### S.5 Interpretation rules

1. **Not supported in either market.** "The real graph is not distinguishable from placebos that also
   preserve the industry–size composition of every edge." This is reported next to the parent result (rank
   17 and 26 of 100).
2. **Supported in a market.** "The real graph beats size-stratified placebos in market m."
   - If the parent edge test in m was not supported, this is reported as a discordance and attributed to
     neither null.
   - Finer placebos are closer to the real graph, so in expectation they make rejection harder, not easier.
3. Either way, the rank of the real graph under both nulls is reported side by side.
4. No result of Study S changes a parent verdict.

### S.6 Platform and determinism check

- The parent RSR runs used the laptop: macOS arm64, `~/.venvs/rsr_arm`, torch 2.8.0, numpy 2.0.2, and 1
  thread. Study S runs on the **same platform** so that common random numbers with the parent's real graph
  hold.
- The re-run of the real graph at seed 0 must reproduce the parent's `{market}_wikidata_real_s0.npz` test
  IRR to within 1e-9.
  - If it does, T(real) is the parent value.
  - If it does not, the mismatch is recorded as a deviation, and T(real) is the re-run value, so that every
    graph in the test comes from the same code path and platform.

### S.7 Descriptive (pre-specified, return-free)

For each market, report the following for the real graph, the parent placebos (k = 0–98) and the
size-stratified placebos (k = 0–98):
- the Spearman correlation between Wiki degree and a_i, as a mean over k
- the mean edge overlap of the placebos with the real non-self edges
- the share of linked stocks that are movable

## Study K — RT-GCN(U): K = 99 placebos (reviewer point 2)

### K.1 Design

- **Cells.** NASDAQ_U (main study) and NYSE_U (extension).
- **New models.** For each cell, placebos k = 19–98 are trained at seed 0, validation arm, primary `paper`
  config. They are generated by the **frozen** runners with the **frozen** RNG stream
  `default_rng([20261111, 7, k])`. The existing k = 0–18 are kept, not re-run and not replaced. Together
  they give **K = 99** placebos (perm000–perm098).
  - NASDAQ_U: `scripts/run_rtgcn_replica.py --mode val --cfg paper --graph perm --k 19:99 --seeds 0`.
    1 thread, output `outputs/rtgcn_v1`.
  - NYSE_U: `scripts/run_rtgcn_ext.py --market NYSE --variant U --mode val --graph perm --k 19:99 --seeds 0`.
    1 thread, the frozen cell count, output `outputs/rtgcn_ext_v1`.
- The frozen parent reports read only `range(19)`, so the parent results are unaffected by the new files.
- Platform: DIPC hyperion, bcam-exclusive, `PyTorch/2.1.2-foss-2022b`. This is the same platform as the
  existing placebos and real models of both cells. The parent freeze records must still verify, and the
  frozen runners enforce this.
- In total there are 160 new models.

### K.2 Hypothesis

**H-C99_c**, for c in {NASDAQ_U, NYSE_U}: the real graph beats within-industry placebos.
- T(G) is the IRR-1 at the validation-selected epoch, seed 0, over all 172 test days. This is identical to
  the parent H-C statistic.
- p = (1 + #{k ≤ 98 : T(P_k) ≥ T(real)}) / 100.
- **Family K: Holm over the 2 cells at α = 0.05.**

**Why extending K after seeing the K = 19 result spends no α.**
- The parent H-C tests were outside every Holm family, and at K = 19 no supported conclusion was possible.
  Hence no decision was taken at the first look.
- The extension is unconditional. It is applied to both U cells whatever their K = 19 p-values, and K is
  fixed here before any new placebo is trained.
- The 99 placebos are i.i.d. draws from the same orbit regardless of when they are trained. So the K = 99
  rank test is exact under the parent null.

### K.3 Interpretation rules

1. **Not supported.** The test is reported as "not detected at K = 99", together with the rank of the real
   graph among 100 graphs and the updated detectable effect (K.4). It is never described as "no relation
   value".
2. **Supported.** "The specific relations of cell c beat within-industry placebos at K = 99." The paper's
   K = 19 statements for that cell are revised accordingly. The parent D1 verdicts are unaffected.
3. The T cells are **not** extended (K.5). Their K = 19 results stand as reported.

### K.4 Secondary and descriptive

- **Detectable effect at K = 99.** This is the shift of the real graph, in placebo standard deviations and
  in IRR-1 units, that gives 80% power for the one-sided rank test at α = 0.05 and at the Holm first step
  (α = 0.025). The same calculation as paper §power is used, now with the observed spread of 99 placebos.
- **NYSE_U without test day 89.** A descriptive sensitivity, as in DEVIATIONS RX-S1.
- **Ranks.** The rank of the real graph among 20 (old) and among 100 (new) graphs.

### K.5 Out of scope, with reasons

- **RT-GCN(T) cells.**
  - The compute cost is about 1,400 core-hours (NASDAQ_T) and about 3,000 core-hours (NYSE_T).
  - The T variant does not reproduce its published level (parent R0).
  - The NYSE test set is dominated by day 89.
  These are stated as limitations.
- **Size-stratified placebos for RT-GCN.** The released RT-GCN data contain no ticker list, and the prices
  are normalised. A size measure would first require recovering tickers by matching return series. That is
  not attempted here, and it is stated as a limitation.

## Execution and completion

| study | cell | new models | platform | threads | est. wall |
|---|---|---|---|---|---|
| S | NASDAQ, NYSE | 2 × (99 + 1) = 200 | laptop, 6 queues | 1 | ~5–6 h |
| K | NASDAQ_U | 80 | hyperion array | 1 (6 GB) | ~1.5 h |
| K | NYSE_U | 80 | hyperion array | 1 (10 GB) | ~4 h |

- **Completion rule (as in the parents).** A family's tests are computed only when all of its models
  exist. A failed model is re-run with the same seed. If that is impossible, the study is reported as
  incomplete, and no test is computed on a subset.
- **Analysis.** `dapilot.rev1_report.summarise(yaml.safe_load(open('configs/rev1.yaml')))` is run once,
  after completion. Its output goes to `RESULTS_REV1.md`.
- **Deviations.** Deviations are recorded in `DEVIATIONS.md` with the prefix R1.

## Dev rehearsal (before the freeze; no test-period data)

1. `tests/test_rev1.py` must pass. It checks the following.
   - Size bins are computed from training dates only.
   - The permutation preserves the industry label and the size bin of every node.
   - The edge count is preserved.
   - The RNG stream is distinct from the parent's.
   - The K-study task list contains exactly k = 19–98 for both cells, with seed 0.
2. RSR dev mode, with no 2017 rows: 2 size-stratified placebos per market at 50 epochs, run through the
   new runner, plus the S.7 descriptives. These are reported here before the freeze.
3. On hyperion, a dry check that the frozen runners still verify their freeze records. One perm019 smoke
   run in dev mode per cell.

### Dev rehearsal results (2026-11-16, before the freeze)

**1. Tests.**
- Laptop: `tests/test_rev1.py` gives 10 passed.
- Hyperion: 7 passed, 3 skipped. The skipped tests need the RSR raw data, which is laptop-only.
- Hyperion also runs the parent suites `tests/test_rtgcn_replica.py` and `tests/test_rtgcn_ext.py`.
  These pass, in the same run as above (SLURM 7995840).

**2. RSR dev mode** (no 2017 rows; the dev "test" is the parent's validation year). These numbers are
disclosed.

| model | ep* | dev val IRR | dev pseudo-test IRR | time |
|---|---|---|---|---|
| NASDAQ sizebin000 / 001 | 39 / 39 | +1.117 / +0.959 | +1.463 / +1.507 | 101 / 95 s |
| NYSE sizebin000 / 001 | 44 / 44 | −0.369 / −0.362 | −0.075 / +0.266 | 225 / 221 s |

- The selected epochs equal those of the parent dev placebos (39 and 44; `PREREG_RSR_REPLICATION.md` §10).
- Nothing was changed because of these runs.

**S.7 descriptives (return-free; `outputs/rev1_rsr_sizebin_dev/descriptives_S7.json`, K = 99):**

| market | Spearman(deg, size): real | parent placebos | size-bin placebos | edge overlap: parent | size-bin | linked movable | cells |
|---|---|---|---|---|---|---|---|
| NASDAQ | 0.390 | 0.186 | 0.325 | 5.7% | 13.0% | 94.8% | 209 |
| NYSE | 0.442 | 0.230 | 0.367 | 5.1% | 11.5% | 96.2% | 237 |

- The parent overlaps reproduce `edge_overlap_s8_6.json` (5.7% and 5.1%), which cross-checks the
  regeneration of the parent placebos.
- The size-bin placebos restore most of the degree–size association that the parent placebos break.
- No stock has a missing size.

**3. Hyperion smoke** (SLURM 7995840, 4.5 min).
- Both parent freeze records verify with the hyperion copy, and the hyperion code hashes equal the laptop's.
- `run_rtgcn_replica.py` dev perm019 (NASDAQ) and `run_rtgcn_ext.py` dev perm019 (NYSE U) ran end to end
  (2 epochs), with outputs in `outputs/rev1_k99_smoke`.
- At freeze time the existing confirm outputs hold perm000–perm018 for both cells, and no perm019+.

**Revised timing.** About 1.7 min (NASDAQ) and 3.8 min (NYSE) per RSR dev model. The confirm models train
on a longer window.

## Known limitations

- The addendum is post hoc in motivation (§0).
- Dollar volume is a proxy for size, and only 2 bins are used.
- The size bins preserve the degree–size association only approximately.
- Study S covers RSR only, and Study K covers the U variant only.
- Each graph is still trained once (seed 0). Multi-seed placebos were considered. They would cost the same
  number of models, but they would not lower the smallest attainable p. So K = 99 was chosen.
