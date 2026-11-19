# Placebo-Calibrated Graph Attribution (PCGA): replication package

Code, pre-registrations, freeze records and results for the paper

> **Placebo-calibrated graph attribution for stock prediction: pre-registered re-tests of two published
> graph neural networks** (submitted to *Expert Systems with Applications*).

PCGA asks two separate questions of a relational stock model:

1. **Graph utility (D1).** Does the model with its graph beat the same model without relations?
   Seed-paired test over retrained models.
2. **Relation specificity (edge).** Does the *real* graph beat structure-preserving placebo graphs
   (within-industry node permutations)? Exact placebo rank test,
   `p = (1 + #{T(placebo_k) >= T(real)}) / (K + 1)`, Holm-corrected across markets.

**PCGA-L** extends this to a placebo ladder (L0 none -> L1 Erdős–Rényi -> L2 degree-preserving ->
L3 within-industry -> L4 real), whose steps (density, degree, industry, edge) telescope to D1.

Every confirmatory analysis was pre-registered and its code and configuration were frozen with SHA-256
hashes **before** the test period was read. `DEVIATIONS.md` logs every departure from the plans.

## Repository layout

The layout is the same as the working project, so that the freeze records verify byte for byte.

| Path | Contents |
|---|---|
| `PREREG_*.md` | Pre-registrations (RSR ladder, RSR re-test, RT-GCN, RT-GCN extension, PCGA-L cross-model, Revision 1) |
| `*FREEZE*.md`, `PCGA_L_MULTIMODEL_FREEZE.json` | SHA-256 freeze records of the code and configs used by each confirmatory run |
| `DEVIATIONS.md` | Deviation log |
| `RESULTS_*.md` | Confirmatory and post hoc results, as reported in the paper |
| `dapilot/` | Library: placebo graphs, PCGA-L ladder, statistics, RSR port, RT-GCN harness, Rev1 size-bin placebos |
| `scripts/` | Runners, report builders, systematic-search scripts, paper figures; `scripts/hyperion/` holds the SLURM files actually used |
| `configs/` | Machine-readable experiment specifications |
| `tests/` | Unit and integrity tests |
| `outputs/` | Per-experiment summaries, reports, JSON and CSV results (raw prediction arrays `*.npz` are not included, see below) |
| `literature/` | Systematic-search protocol, query records, screened pools, coding sheets, codebook and inter-coder results |

## Paper → files

| Paper section | Pre-registration / freeze | Results | Runner |
|---|---|---|---|
| Construct check (§3.6) | `BENCHMARK_FREEZE.md` | `outputs/da_synthetic_pilot_validity_v2/report.md` | `scripts/run_pilot.py` |
| PCGA-L cross-model recovery (§3.6) | `PREREG_PCGA_L_MULTIMODEL.md`, `PCGA_L_MULTIMODEL_FREEZE.json` | `RESULTS_PCGA_L_MULTIMODEL.md` | `scripts/run_pcga_l_multimodel.py` |
| RSR re-test (§4) | `PREREG_RSR_REPLICATION.md`, `PREREG_REPLICATION_FREEZE.md` | `RESULTS_RSR_REPLICATION.md`, `outputs/rsr_replica_v1/` | `scripts/run_rsr_replica.py` |
| RT-GCN re-test (§5) | `PREREG_RTGCN.md`, `PREREG_RTGCN_FREEZE.md` | `RESULTS_RTGCN.md`, `outputs/rtgcn_v1/` | `scripts/run_rtgcn_replica.py` |
| RT-GCN extension (§5) | `PREREG_RTGCN_EXT.md`, `PREREG_RTGCN_EXT_FREEZE.md` | `RESULTS_RTGCN_EXT.md`, `outputs/rtgcn_ext_v1/` | `scripts/run_rtgcn_ext.py` |
| Literature audit (§6) | `literature/SYSTEMATIC_SEARCH_PROTOCOL.md` | `literature/SYSTEMATIC_SEARCH_RESULTS.md`, `RESULTS_RQ2_INTERCODER.md` | `scripts/syssearch_*.py`, `scripts/rq2_intercoder_reliability.py` |
| PCGA-L with a ridge expert (§7, Suppl. S1) | `PREREG_RSR.md`, `PREREG_FREEZE.md` | `RESULTS_RSR.md`, `outputs/power_ladder_v2/` | `scripts/run_power_ladder_v2.py` |
| Size-matched placebos, RT-GCN K=99 (§8.2) | `PREREG_REV1.md`, `PREREG_REV1_FREEZE.md` | `RESULTS_REV1.md`, `outputs/rev1_results.json` | `scripts/run_rev1_rsr_sizebin.py`, `scripts/hyperion/rev1_*.sbatch`, `dapilot/rev1_report.py` |
| Power, sub-periods, day 89 (§8.3, Suppl. S3) | — (post hoc, labelled as such) | `RESULTS_POSTHOC_REVIEW.md` | `scripts/power_posthoc.py`, `scripts/posthoc_subperiods_equivalence.py`, `scripts/rtgcn_day89_check.py` |
| Figures | — | — | `scripts/paper_figures.py` |

## Verifying the freeze records

```
python scripts/verify_freezes.py
```

Every frozen file that is shipped here must match its recorded hash. Files that are not shipped (upstream
data and code, prediction arrays) are reported as absent. `PREREG_RTGCN_FREEZE_v0.1_superseded.md` is
a superseded record, kept for transparency, and is expected not to match (see `DEVIATIONS.md`).

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_upstream.sh          # clones RSR and RT-GCN at the frozen commits
```

The upstream data and pretrained embeddings are hosted by their authors and are not redistributed here;
see [`data/README.md`](data/README.md) for the exact sources, commits and checksums.

```
python -m pytest -q tests
```

Without any upstream material, the method and statistics tests run on their own:

```
python -m pytest -q tests --ignore=tests/test_rtgcn_replica.py --ignore=tests/test_rtgcn_ext.py
```

(69 pass; the two RSR data-loading tests need `data/raw/rsr_repo`.) With all upstream code and data in
place, the full suite has 90 tests. The tests themselves are part of the frozen material and are
therefore not changed to skip when data is missing.

## Reproducing

Each runner has a `dev` mode, which never reads the test period, and confirmatory modes. The RSR
re-test and PCGA-L runners refuse to run in confirmatory mode unless their freeze record verifies. For
the other studies, run `scripts/verify_freezes.py` first. Examples:

```
python scripts/run_rsr_replica.py --mode confirm --market NASDAQ --graph real --seeds 0:20
python scripts/run_rsr_replica.py --mode confirm --market NASDAQ --graph perm --k 0:19
python scripts/run_rsr_replica.py --summarise

python scripts/run_rtgcn_replica.py --mode val --cfg paper --graph perm --k 0:19
python scripts/rtgcn_summarise.py

python scripts/run_rev1_rsr_sizebin.py --mode confirm --market NYSE --graph sizebin --k 0:99
python -c "import yaml; from dapilot.rev1_report import summarise; summarise(yaml.safe_load(open('configs/rev1.yaml')))"

python scripts/run_pcga_l_multimodel.py --mode confirm --all --jobs 8
python scripts/run_pcga_l_multimodel.py --summarise
```

The exact seeds, placebo indices and model settings of every confirmatory run are fixed in the
corresponding `configs/*.yaml` and pre-registration. The full confirmatory set is about 900 trained
models; the RT-GCN and Revision 1 runs were executed on a SLURM CPU cluster with the job files in
`scripts/hyperion/`.

The summarisers read the per-model prediction arrays (`outputs/**/*.npz`, about 1.3 GB), which are too
large for git. They will be deposited in a public archive with a DOI; until then they are available on
request.

## Software

Python 3.9–3.12, PyTorch 2.x, NumPy, SciPy, pandas, scikit-learn, PyYAML, Matplotlib (versions in
`requirements.txt`). The RSR runs use the authors' frozen pretrained embeddings, so no TensorFlow
training is needed. TensorFlow 2.15 is needed only for `scripts/run_rsr_original_dev.py`, a development
check that runs the authors' original TF code.

## Licence

Code in this repository is released under the MIT licence (`LICENSE`), except
`dapilot/rsr_replica.py`, a port of the AGPL-3.0 RSR code of Feng et al. (2019), which is released under
the GNU AGPL-3.0 (`LICENSES/AGPL-3.0.txt`). Documents (pre-registrations, results, coding sheets) are
released under CC BY 4.0. Upstream code and data remain under their authors' terms.

## Citation

See `CITATION.cff`.
