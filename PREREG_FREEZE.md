# Pre-registration freeze record — RSR open-track ladder

**Frozen:** 2026-11-09 18:10:35 CET
**Frozen by:** hbukhori@bcamath.org
**Data status at freeze:** no RSR data downloaded, opened, or inspected in this workspace. No RSR
files exist locally.

## Frozen files (sha256)

| File | sha256 | Role |
|---|---|---|
| `PREREG_RSR.md` | `26d8479c25eb20e95e1b6ef63e1b00da704ab6075f5fdda1244f67fadc5e39e6` | pre-registration (v0.2) |
| `configs/rsr_prereg.yaml` | `de354ae42f1da5bffd79c8f9ec02c649feead0731b746571b15c89f806bf441d` | machine-readable spec |
| `dapilot/power.py` | `f434f0312eb6992b8223cf8b66b5c9a4889d49d6eff0ca84fa9c82e29de48fa7` | placebo swap algorithms, ridge expert, block bootstrap |
| `dapilot/power_v2.py` | `382dfcf4e0b51999bacf93d364e13986c814980e598b1a0bf9809771e2c25dfe` | placebo-rank test |

## Upstream evidence (sha256, unchanged since the power-analysis decision)

| File | sha256 |
|---|---|
| `BENCHMARK_FREEZE.md` | `5f8a5a465954895684c3f219c9c1f7c31834b779adc5764de8988551dd7c6846` |
| `configs/benchmark_freeze.yaml` | `b1e804cd7b1839455eeee7c56e75ea12ba7ddcda1f28801e38b7dfe7072942e7` |
| `configs/power_ladder_v2.yaml` | `554acbb1d3b808510545bfbac242234392d4b7e43f3482846ba1811caf6caf89` |
| `configs/power_ladder_v2_1.yaml` | `cf691a8c776ac32611b7f1f40c8e59a0f6b3ce37406c61b282a5a81392b772d7` |
| `outputs/power_ladder_v2/size_and_selection.csv` | `76bb8dbbc8ad541fbcf8b97777245d16b4907b351054a3c3abfe62b801e0d9f6` |
| `outputs/power_ladder_v2/power.csv` | `98267e47e47f731a7dc9cab89df0d160f2eb13520e1c46eceb85e57325e0c26e` |
| `outputs/power_ladder_v2/REPORT.md` | `3ba7d3ed3d9479547a9a3226c8d1c91e1ccb0bfa50d0b4dc1292d97b40c92da1` |

## Summary of what is locked

- Confirmatory family R1–R4: {D1, edge} × {NASDAQ, NYSE}, with Holm at α = 0.05.
- D1: circular block bootstrap with L = 20. Edge: placebo-rank test with K_L3 = 99.
- L4 = Wiki relations. L3 placebos preserve degree and same-industry-neighbour count.
- Primary expert: ridge on `graph_features`.
- Deviation D-K (K = 19 → 99) and amendment A1 (Wiki added to the freeze graph list) were both
  decided before any data were seen.

## Rules after the freeze

1. The frozen files above are not edited. To verify:
   `shasum -a 256 PREREG_RSR.md configs/rsr_prereg.yaml dapilot/power.py dapilot/power_v2.py`.
2. New code (data loader, RSR pipeline) goes in new files. If `dapilot/power.py` or
   `dapilot/power_v2.py` must change, the change and its reason are recorded in `DEVIATIONS.md`
   before the test period is read.
3. Every deviation goes in `DEVIATIONS.md`, with a timestamp, the reason, and whether it was made
   before or after test data were read.
4. The test period (2017) is read only once, after the pipeline has passed tests on train and
   validation.

**External copy:** the user stores a copy of this file outside the workspace (e.g. OSF or email to
self). Record the location and date of that copy here: ____________________
