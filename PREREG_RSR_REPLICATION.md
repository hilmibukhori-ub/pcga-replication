# Pre-registration — re-test of the published RSR relation claim (v0.2)

**v0.2 (2026-11-10, before freeze):** the D1 test was changed and the seed counts were set after
the dev rehearsal (§10). User decision: edge uses 1 seed per graph.
**Status:** FROZEN 2026-11-10 (see `PREREG_REPLICATION_FREEZE.md`).
Confirm mode in `scripts/run_rsr_replica.py` refuses to run until the hashes in that file match.
**Drafted:** 2026-11-10. Machine-readable spec: `configs/rsr_replica.yaml`.
Code: `dapilot/rsr_replica.py`, `dapilot/rsr_replica_infer.py`, `dapilot/rsr_replica_report.py`,
`scripts/run_rsr_replica.py`. Tests: `tests/test_rsr_replica.py`. Fidelity and rehearsal tools (not used
in confirm mode): `scripts/run_rsr_original_dev.py`, `scripts/rsr_rehearsal_analysis.py`.
**Environment:** `~/.venvs/rsr_arm` (native arm64, Python 3.9.6, torch 2.8.0, numpy 2.0.2), on CPU with
one thread per process. The x86 Anaconda Python runs under Rosetta, and its numbers agree to about 1e-9.

## 0. Disclosure of what has been seen

- **The 2017 test period has already been read**, once, by our own confirmatory ridge analysis
  (`RESULTS_RSR.md`, `outputs/rsr_ladder_v1`), and by the exploratory GCN run
  (`outputs/rsr_gnn_v1`). Those runs used different features and models: our own 4 return
  features, ridge and GCN.
- **No output of the RSR model (or of its placebos) on 2017 has been computed** before the freeze.
- Before the freeze, the RSR port was run only in **dev mode**. Dev mode truncates the data at
  index 1008, which removes every 2017 row, and shifts the split one year back. Its results are
  reported in §10.
- The published results (Feng et al. 2019, Tables 5–7) are known. They are the claim being
  tested.

## 1. Claim under test

Feng, He, Wang, Luo, Liu & Chua (2019), *Temporal Relational Ranking for Stock Prediction*, ACM
TOIS 37(2). arXiv 1809.09441, saved as `literature/feng2019_rsr_1809.09441.pdf` (sha256
`735675e8…386f`). Table 7, Wiki relations, mean ± sd over 5 runs, test period 2017:

| | NASDAQ IRR | NYSE IRR |
|---|---|---|
| Rank_LSTM (no relations) | 0.68 ± 0.60 | 0.56 ± 0.68 |
| RSR_I (Wiki) | 1.19 ± 0.55 | 0.79 ± 0.34 |

The text says that all methods using Wiki relations outperform Rank_LSTM "with a significant
improvement (>0.09) w.r.t. IRR on NYSE", and that the results "verif[y] the merit of encoding
stock relations". No statistical test is reported.

We test two things:

- **(a) D1, the published claim.** RSR_I with Wiki relations has a higher IRR than the same
  model without relations.
- **(b) edge, our extension.** The *specific* Wiki relations beat placebo relations that
  preserve the complete typed relation structure and the industry composition of every edge
  endpoint.

## 2. Data, code and provenance

- Repository `fulifeng/Temporal_Relational_Stock_Ranking`, commit `cfbb01bd…2a82d` (intake:
  `outputs/rsr_intake/INTAKE.md`). EOD files `data/2013-01-01`, tickers files, and relation
  tensors from `relation.tar.gz`:
  - wiki: NASDAQ `00b0df3a…4d40`, NYSE `42416de7…9c22`
  - industry: NASDAQ `4838497b…e3fd`, NYSE `41ab68f2…6883`
- **Pretrained sequential embeddings**, from the authors' Google Drive link in the README
  (id `1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT`), downloaded 2026-11-10 to `data/raw/rsr_pretrain_dl/`.
  - archive sha256 `205973c7…acd0`
  - `NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy` (1026×1245×64) `071a13ad…01b2`
  - `NYSE_rank_lstm_seq-8_unit-32_0.csv.npy` (1737×1245×32) `c848999b…f6d8`
- Split: the original code's indices. Train offsets use labels up to index 756 + seq (as in the
  original), validation labels fall in [756, 1008), and test labels in [1008, 1245), which is 237
  days.

## 3. Model (faithful port of `training/relation_rank_lstm.py`, default "sum weight" = RSR_I)

The port is PyTorch, and the relational layer is implemented sparsely. A test checks it against
the dense TF formula on the real Wiki tensor. Everything else mirrors the original:

- data semantics
- glorot/zero initialisation
- leaky-ReLU with α = 0.2
- column-wise softmax with the −1e9 mask, including the self-relation channel
- return-ratio output
- loss = weighted MSE + α · pairwise rank loss
- Adam with lr 1e-3, 50 epochs, and the full cross-section of one day per batch
- epoch selection by the lowest validation total loss

The pretrained embeddings stay fixed, as in the original.

| | seq | units | α (rank loss) | source |
|---|---|---|---|---|
| NASDAQ | 16 | 64 | 0.1 | README command (`-rn wikidata`) |
| NYSE | 8 | 32 | 10 | README command. That command uses the default relation (industry). It is the only published NYSE setting, and the only NYSE embedding uses seq 8 / units 32, so it is used for Wiki as well. **This is an assumption.** |

**No-relation baseline (L0):** the same model, with the relation tensor reduced to the
self-relation channel. The propagated vector then equals the node's own embedding, so this is the
paper's "remove the relational embedding layer" (Rank_LSTM) on the same frozen embedding. The
paper's Rank_LSTM was trained end to end, which is a disclosed difference.

**Seeds:** model seed = 20261110 + 1000·market + s. D1 uses s = 0…19 for real and L0 (the paper
used 5 runs, and seeds 0–4 are also reported). Edge uses s = 0 for every graph. Industry
(secondary) uses s = 0…4. The same seeds are used for every
graph (common random numbers). The seed sets the initialisation and the per-epoch order of
training days. The paper averages 5 runs.

## 4. Placebo relations

**Within-industry node permutation.** π is a uniformly random permutation that maps each stock to
a stock with the same industry label, with the `n/a` label pooled as one group. The placebo
tensor is R'[i, j, :] = R[π(i), π(j), :].

It preserves exactly:
- the typed, directed multigraph up to relabelling: every relation type, every edge direction,
  the self channel, and the degree sequence within each industry
- the industry label of both endpoints of every edge

It changes only *which stock within an industry* holds which relations. Stock embeddings stay
attached to their stocks.

Coverage: 113 (NASDAQ) and 130 (NYSE) groups. 17 and 23 stocks are alone in their group and stay
fixed. No edge connects two fixed stocks.

**Exactness:** under H0, where the within-industry assignment of relations carries no
information about the test metric, the real tensor is one draw from the permutation orbit. It is
then exchangeable with the K placebos, so the rank test in §6 is exact for any K. This argument
is stronger than the one for the swap placebos in `PREREG_RSR.md` (P1b).

**K = 99** for each market, generated with `default_rng([20261110, market, 7, k])`.

The industry relation tensor is invariant under this permutation, so no edge test is defined for
it (§8).

## 5. Metrics

- **Primary: IRR = Σ over the 237 test days of the realised return of the top-1 predicted stock.**
  This is the paper's main metric, as defined in its text. The code's `btl` equals 1 + IRR, and
  both are reported.
- Secondary: MSE (pooled over stock-days, as in `evaluator.py`), MRR of the top-1 pick, and
  mean top-5 and top-10 returns.

## 6. Confirmatory hypotheses (Wiki relations)

| ID | Market | H1 | Test |
|---|---|---|---|
| H1 | NASDAQ | D1-IRR > 0 | one-sided **paired t-test over seeds 0–19**: IRR(real, s) − IRR(L0, s) |
| H2 | NYSE | D1-IRR > 0 | same |
| H3 | NASDAQ | edge-IRR > 0 | placebo rank: T(G) = IRR at seed 0; p = (1 + #{k : T(P_k) ≥ T(real)}) / 100 |
| H4 | NYSE | edge-IRR > 0 | same |

**Holm across H1–H4 at α = 0.05.** The smallest possible edge p is 0.01, which is below the
smallest Holm threshold of 0.0125.

**Verdicts:**
- **supported:** Holm-adjusted p ≤ 0.05.
- **not detected:** otherwise. It is never described as "no effect".

**Why a paired t-test (v0.2):** in the dev A/A check (§10), the v0.1 day bootstrap rejected
8–11% of the time at a nominal 5%. The seed t-test rejected 4.0–4.5%. The t-test conditions on the
test period. It answers the question the paper's "mean ± sd over 5 runs" addresses: over training
randomness, does real beat L0 on this test set? Generalisation across days is reported as a
secondary result: a two-way seed × day-block bootstrap (L = 20, 2,000 reps), which was
conservative in A/A (1–1.5%). The day-only bootstrap is reported only as descriptive and is
labelled as miscalibrated. The edge test is exact and does not depend on any of this.

## 7. Interpretation rules (fixed in advance)

1. **If H1/H2 are not detected:** the published claim that Wiki relations improve IRR does not
   replicate under a matched baseline and a seed-averaged statistical test. The point estimates
   are reported next to the paper's.
2. **If H3/H4 are supported:** the specific Wiki relations carry information beyond typed
   structure and industry composition. Wiki relations are not point-in-time, so the look-ahead
   caveat applies.
3. **If H1/H2 are supported but H3/H4 are not:** the gain is attributed to relation structure or
   industry composition, not to the specific relations. This is reported as "the claim
   replicates, but its mechanism is not the specific relations".
4. **Reproduction check R0 (descriptive):** if our real-Wiki mean IRR over seeds 0–4 differs from
   the paper's by more than the paper's reported sd, this is reported as "reproduction gap". Port
   fidelity against the original TF code was checked separately on dev data (§10). The analysis is
   not changed or retuned.

## 8. Secondary and descriptive analyses (not confirmatory)

1. R0: our mean ± sd over seeds 0–4 (and over all 20) against Tables 5 and 7 (IRR, MSE, MRR), for real and L0.
2. **Claim-threshold check:** the share of the 99 placebos whose IRR (seed 0) exceeds L0's by
   more than 0.09, the margin the paper calls significant.
3. Edge and D1 on MSE and MRR (rank p and bootstrap p, descriptive).
4. D1 for the industry relation (RSR_I-industry against L0 seeds 0–4, paired t, both markets),
   next to Table 6.
7. D1 secondaries: the two-way bootstrap p; the 5-seed paired t (seeds 0–4, the paper's run
   count); the day-only bootstrap p, labelled miscalibrated; and D1 on daily MSE.
5. Per-seed spread of IRR, the distribution of placebo IRR, and best epochs.
6. Mean edge overlap between the placebos and the real relations.

## 9. Execution

1. Dev sanity runs (no 2017 rows) for real, L0 and two placebos per market, at 50 epochs. The
   report code is smoke-tested on the dev output with reduced seeds and K.
2. Freeze: sha256 of this document, the config, the code and the tests, recorded in
   `PREREG_REPLICATION_FREEZE.md`.
3. Confirm run, in 6 parallel queues:
   - Stage A first in every queue: real and L0 with seeds 0–19, and industry with seeds 0–4.
   - Stage B follows automatically in the same queues: 99 placebos at seed 0.
   - That is 144 models per market, at roughly 6 and 12 minutes per model for NASDAQ and NYSE on
     arm64, or about 7–8 h wall in total.
   - Both stages are committed. Stage B runs whatever stage A shows.
4. `python scripts/run_rsr_replica.py --summarise`. Deviations are recorded in `DEVIATIONS.md`
   with the prefix RP.

## 10. Dev sanity results (pre-freeze, no 2017 data)

Dev mode: train offsets < 504, validation labels in [504, 756), pseudo-test labels in [756, 1008)
(= the 2016 validation year of the original split). There is one seed (s = 0) and 50 epochs, and no
2017 row is loaded. Logs are in `outputs/rsr_replica_dev/logs`.

| Model | best epoch | pseudo-test IRR | MRR | MSE |
|---|---|---|---|---|
| NASDAQ L0 (no relation) | 39 | +1.665 | 0.0375 | 4.953e-4 |
| NASDAQ real Wiki | 39 | +1.644 | 0.0371 | 4.954e-4 |
| NASDAQ placebo 0 / 1 | 39 / 39 | +1.423 / +1.591 | 0.0353 / 0.0367 | 4.954e-4 |
| NYSE L0 | 44 | +0.898 | 0.0575 | 3.685e-4 |
| NYSE real Wiki | 44 | +0.335 | 0.0485 | 3.673e-4 |
| NYSE placebo 0 | 44 | −0.152 | 0.0491 | 3.671e-4 |

What these runs show:
- Training is stable. With common random numbers, every model in a market selects the same epoch.
- The single-seed IRR differs by up to about 1.0 between graphs that are almost identical in MSE.
  This is the noise level that the seed averaging and the placebo-rank test are meant to handle.

These are single-seed results on a different year. They are not evidence about the claim.

### 10b. Dev rehearsal, NASDAQ (v0.2 basis; `outputs/rsr_replica_dev/rehearsal_NASDAQ.txt`)

The runs were L0 at seeds 0–19, real Wiki at seeds 0–4, and 19 placebos at seed 0. The
pseudo-test year is 2016, and no 2017 row was loaded.

**Fidelity against the original TF code.** `scripts/run_rsr_original_dev.py` runs
`relation_rank_lstm.py` unchanged under tf.compat.v1 (TF 2.15, arm64), with the same dev
truncation and seeds 123456789, 1 and 2:

| | test IRR per seed | mean ± sd | test MSE | test MRR | val MSE |
|---|---|---|---|---|---|
| Original TF (3 seeds) | 2.18, 1.69, 1.30 | 1.72 ± 0.44 | 4.928–4.954e-4 | 0.026–0.031 | 4.579–4.584e-4 |
| Port (5 seeds) | 1.64, 1.03, 2.20, 1.40, 2.00 | 1.65 ± 0.47 | 4.931–4.954e-4 | 0.025–0.037 | 4.574–4.586e-4 |

The distributions match. NYSE fidelity was not run separately: the code path is identical, and
only the hyperparameters differ.

**Seed noise.**
- L0 IRR over 20 seeds: mean 1.64, sd 0.54. This is similar to the paper's ±0.55–0.60.
- Day-bootstrap SE of a single model's IRR: 0.63.
- Correlation of per-seed IRR between real and L0 under common random numbers: +0.78 (5 seeds).

**A/A calibration of the D1 tests.** These are L0-vs-L0 comparisons on disjoint seed sets, 200
random splits of the 20 seeds. Splits overlap, so the rates are approximate.

| test | 5 vs 5 | 10 vs 10 | nominal |
|---|---|---|---|
| day-only block bootstrap (v0.1) | 0.110 | 0.080 | 0.05 |
| seed t-test | 0.040 | 0.045 | 0.05 |
| two-way seed × day bootstrap | 0.010 | 0.015 | 0.05 |

The paired t-test was therefore made primary, and the two-way bootstrap secondary (§6).

**Placebos (seed 0, n = 19).** IRR mean 1.45, sd 0.19, range 0.99–1.74. The spread across graphs at
a fixed seed is much smaller than the seed sd, because common random numbers absorb most of the
seed noise. This motivated the user's choice of 1 seed for the edge test. The test stays exact.

**What was seen and what was changed.** The real-vs-L0 and real-vs-placebo comparisons on the dev
year were visible: real IRR 1.65 vs L0 1.69 (seeds 0–4), and real at seed 0 above 16 of the 19
placebos. The v0.2 changes come from the A/A calibration (the D1 test), the fidelity check, and
compute time on arm64 (the seed counts). None of them was chosen to favour any outcome.

**Report code** was smoke-tested on the dev outputs.

## 11. Known limitations

- Fixed embeddings: this tests the relational layer on the authors' embedding, not end-to-end
  training.
- The NYSE α is taken from the README's industry command (§3).
- The port is not bit-identical to TF1 (RNG, kernels). Fidelity is checked by unit tests and by
  R0, not by rerunning TF1.
- One test year (237 days). IRR is dominated by single-stock daily returns, which the paper
  itself notes.
- The Wiki and industry relations are not point-in-time.
