# RSR re-test (Feng et al., TOIS 2019): RSR_I with Wiki relations, NASDAQ and NYSE. Confirmatory results

Pre-registration: `PREREG_RSR_REPLICATION.md` v0.2 (frozen 2026-11-10 10:55; hashes in
`PREREG_REPLICATION_FREEZE.md`). Runs: laptop (arm64, `~/.venvs/rsr_arm`), 288/288 models
(per market: 20 real + 20 none + 5 industry + 99 placebos), all outputs load without error.
Analysis: `python scripts/run_rsr_replica.py --summarise` → `outputs/rsr_replica_v1/REPORT.md` and
`results.json` (run 2026-11-10 ~19:30). Primary metric: IRR = sum over the 237 test days (2017) of the
realised return of the top-1 predicted stock. Execution note: RP1 in `DEVIATIONS.md`.

## Confirmatory family (Holm over 4, α = 0.05)

| test | estimate (IRR) | 95% CI | p | p (Holm) | seeds > 0 | verdict |
|---|---|---|---|---|---|---|
| **H1** D1 NASDAQ: Wiki vs none (paired t, 20 seeds) | **+0.119** (0.70 vs 0.58) | [−0.029, +0.267] | 0.054 | 0.217 | 13/20 | not detected |
| **H2** D1 NYSE | **−0.070** (1.27 vs 1.34) | [−0.253, +0.113] | 0.783 | 0.783 | 9/20 | not detected |
| **H3** edge NASDAQ: Wiki vs 99 within-industry placebos (exact rank, seed 0) | **+0.094** (−0.39 vs placebo mean −0.48) | — | 0.17 | 0.51 | 16/99 placebos ≥ real | not detected |
| **H4** edge NYSE | **+0.211** (2.22 vs placebo mean 2.01) | — | 0.26 | 0.52 | 25/99 placebos ≥ real | not detected |

D1 CIs are t-intervals on the 20 per-seed paired differences. Edge estimates are real minus placebo mean,
at seed 0 with common random numbers.

## Interpretation (by the pre-registered rules in §7)

- **Rule 1 applies to both markets.** The published claim that Wiki relations improve IRR does not
  replicate under a matched no-relation baseline and a seed-averaged test.
  - The paper's gains are **outside our 95% CIs**: +0.51 on NASDAQ (CI upper +0.27) and +0.23 on NYSE
    (CI upper +0.11). The paper's baseline is an end-to-end Rank_LSTM, ours is the same frozen embedding
    with the relational layer reduced to the self channel (§3), so the comparison is close but not
    identical.
  - On NYSE the point estimate is **negative**: the no-relation model is ahead in 11 of 20 seeds.
  - On NASDAQ the estimate is positive and the unadjusted p is 0.054, but it is not significant even
    before Holm and is less than a quarter of the paper's gain.
- **Rules 2/3 do not apply.** The specific Wiki relations are not better than within-industry
  placebos that keep the full typed relation structure and the industry of every edge endpoint. The
  real graph sits at rank 17/100 (NASDAQ) and 26/100 (NYSE) from the top.
- Nothing is supported, so the look-ahead caveat for Wiki (rule 2) is moot.
- "Not detected" is not "no effect". The D1 CIs exclude the paper's effect sizes but do not exclude
  small positive effects on NASDAQ.

## Reproduction check R0 (descriptive)

| | ours, seeds 0–4 | ours, 20 seeds | paper (5 runs) |
|---|---|---|---|
| NASDAQ RSR_I-Wiki IRR | 0.59 ± 0.67 | 0.70 ± 0.49 | 1.19 ± 0.55 |
| NASDAQ no relation IRR | 0.46 ± 0.81 | 0.58 ± 0.61 | 0.68 ± 0.60 (Rank_LSTM) |
| NYSE RSR_I-Wiki IRR | 1.10 ± 0.72 | 1.27 ± 0.61 | 0.79 ± 0.34 |
| NYSE no relation IRR | 1.27 ± 0.99 | 1.34 ± 0.71 | 0.56 ± 0.68 (Rank_LSTM) |
| MSE (Wiki), NASDAQ / NYSE | — | 3.791e-4 / 2.261e-4 | 3.79e-4 / 2.26e-4 |
| MRR (Wiki), NASDAQ / NYSE | — | 0.036 / 0.046 | (Table 7) |

- **NASDAQ: reproduction gap** (rule 4). Our seeds-0–4 mean is more than one paper sd below the
  paper's 1.19. MSE matches to four digits.
- **NYSE: within one paper sd.** Our levels are higher than the paper's for both arms, and the
  ordering is reversed (no relation ≥ Wiki).
- Port fidelity against the unchanged original TF code was checked before the freeze on dev data
  (prereg §10b: IRR 1.72 ± 0.44 original vs 1.65 ± 0.47 port, matching MSE). The NASDAQ gap is
  therefore not explained by a porting error we could detect. Possible sources: the authors'
  5 runs, a different seed set, or the paper's end-to-end training.

## Secondary and descriptive results (not confirmatory)

- **D1, other tests** (prereg §6, §8.7):
  - two-way seed × day-block bootstrap: p = 0.22 (NASDAQ), 0.67 (NYSE)
  - 5-seed paired t (the paper's run count): +0.13, p = 0.15 (NASDAQ); −0.17, p = 0.77 (NYSE)
  - day-only bootstrap (**miscalibrated**, A/A 8–11%): p = 0.135, 0.817
  - daily MSE: NASDAQ −2.0e-8 (90% CI [−7.1e-8, +2.9e-8], p 0.74); NYSE −4.0e-8 ([−1.3e-7, +4.5e-8], p 0.79)
- **Claim-threshold check** (§8.2): share of placebos whose seed-0 IRR beats no-relation by more than
  the paper's "significant" margin of 0.09 is 17% on NASDAQ and **0%** on NYSE. On NYSE, the seed-0
  no-relation model (IRR 3.02) beats every one of the 99 placebos and the real graph.
- **Edge on other metrics** (§8.3), rank p: MSE 0.50 (NASDAQ), 0.35 (NYSE); MRR 0.25, 0.39.
- **Industry relations** (§8.4, 5 seeds, against Table 6): RSR_I-industry vs none, −0.10 (p 0.60,
  NASDAQ) and −0.38 (p 0.74, NYSE). No support for the industry relation either.
- **Placebo distribution** (§8.5), seed 0: NASDAQ mean −0.48, sd 0.21, range [−0.79, +0.57];
  NYSE mean 2.01, sd 0.40, range [0.56, 3.01]. Graph-to-graph spread at a fixed seed is well below the
  between-seed sd (0.5–0.7), as in the dev rehearsal.
- **Best epochs** (Wiki): 22–49 (NASDAQ), 28–49 (NYSE). With common random numbers, models at the same
  seed mostly select the same epoch.
- **Edge overlap, placebo vs real** (§8.6). This was computed after the summary and uses no test data
  (`scripts/rsr_replica_edge_overlap.py` → `outputs/rsr_replica_v1/edge_overlap_s8_6.json`).
  - Placebos regenerated exactly as in the confirm run.
  - Overlap is measured as the share of real non-self edges that are also present in a placebo.
  - NASDAQ: 1,937 real edges, mean overlap **5.7%** (range 2.9–11.8%), typed overlap 5.0%.
  - NYSE: 8,468 real edges, mean overlap **5.1%** (range 3.1–8.5%), typed overlap 4.1%.
  - So the placebos keep the typed structure and the industry of every endpoint, yet are about 95%
    disjoint from the real edges. The edge test therefore contrasts substantially different graphs. A
    null result is not explained by placebos that are near-copies of the real graph.

## Determinism check (incidental)

Seven models were trained twice by separate queues (RP1). All seven duplicate runs reproduce the same
selected epoch and identical validation and test metrics, so the confirm runs are deterministic given
the seed.

## Caveats

- The relational layer is tested on the authors' frozen pretrained embeddings, not with end-to-end
  training (prereg §11).
- The NYSE rank-loss α (10) comes from the README's industry command; it is the only published NYSE setting.
- One test year (237 days). IRR is a sum of single-stock daily returns and is very noisy: the seed sd
  of a single model's IRR is about 0.5–0.7, the same size as the paper's claimed gains.
- Wiki and industry relations are not point-in-time.
- The 2017 test period had already been read by our own ridge and GCN analyses (`RESULTS_RSR.md`), with
  different features and models, before this freeze (prereg §0).

## Combined reading with the other re-tests

- RT-GCN(U) NASDAQ (`RESULTS_RTGCN.md`): the published level reproduces, but the relation gain does not
  (H-B −0.47; edge p 0.45).
- Our own ridge/GCN ladder on the RSR data (`RESULTS_RSR.md`): only the NYSE edge step was supported
  under ridge (Holm 0.040), with a tiny effect; it did not survive under GCN.
- Here: none of the four tests supported. Across two published models and two markets, the claimed
  benefit of the relation graph is not detected under matched baselines, multiple seeds and
  placebo graphs.
