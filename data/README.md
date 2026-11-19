# Data

No upstream data is redistributed here. Everything below is published by the original authors. Put it
under `data/raw/` at the paths shown, which are the paths the code and the freeze records use. The
SHA-256 of every file is listed in the freeze record named in each section, and
`python scripts/verify_freezes.py` checks them once the files are in place.

## RSR (Feng et al., 2019, ACM TOIS)

Source: <https://github.com/fulifeng/Temporal_Relational_Stock_Ranking>, commit
`cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d` (`bash scripts/fetch_upstream.sh`).

| Path | What | From |
|---|---|---|
| `data/raw/rsr_repo/` | Repository: EOD prices (`data/2013-01-01`), ticker lists, `relation.tar.gz`, original TF code | git clone |
| `data/raw/rsr_relation/relation/{wikidata,sector_industry}/` | Wiki and industry relation tensors | `tar -xzf data/raw/rsr_repo/data/relation.tar.gz -C data/raw/rsr_relation` |
| `data/raw/rsr_pretrain_dl/pretrain/` | Authors' pretrained Rank_LSTM embeddings (`NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy`, `NYSE_rank_lstm_seq-8_unit-32_0.csv.npy`) | Google Drive link in the RSR README (file id `1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT`) |

Checksums: `PREREG_REPLICATION_FREEZE.md`, `PREREG_REV1_FREEZE.md`; intake record:
`outputs/rsr_intake/`.

## RT-GCN (Zheng et al., 2023, ICDE)

Source: <https://github.com/zhengzetao/RTGCN>, commit `617c3f53a58753251929a0cdbd1c12f7e3df8154`
(`bash scripts/fetch_upstream.sh`). The model code is imported unchanged from `data/raw/rtgcn_repo`.

| Path | What | From |
|---|---|---|
| `data/raw/rtgcn_repo/` | Original model code | git clone |
| `data/raw/rtgcn_dl/NASDAQ/` | `{train,test}_15_{EOD,price,label}`, `NASDAQ_relation.npy` | Authors' Google Drive folder linked in the RT-GCN README, `data/NASDAQ` |
| `data/raw/rtgcn_dl/NYSE/` | Same files for NYSE, plus `NYSE_wiki_relation.npy`, `NYSE_industry_relation.npy` | Same folder, `data/NYSE` |

The released labels are Python pickles. Convert them with `python scripts/rtgcn_convert_labels.py`,
which checks the pickle opcodes against an allow-list and loads them with a restricted unpickler, so
no arbitrary code from the download is executed.

`industry_labels_derived.npy` is the stock-level industry label used for within-industry placebos. It
is read off the diagonal membership of the industry relation channels (edges equal same-label pairs;
stocks with no industry get label -1 and form one permutation group). The NASDAQ file is shipped in
`data/derived/NASDAQ/`; copy it to `data/raw/rtgcn_dl/NASDAQ/`. Its hash, and that of the NYSE file,
is in `PREREG_RTGCN_FREEZE.md` / `PREREG_RTGCN_EXT_FREEZE.md`.

Checksums: `PREREG_RTGCN_FREEZE.md`, `PREREG_RTGCN_EXT_FREEZE.md`.

## Prediction arrays

The per-model test predictions (`outputs/**/*.npz`, about 1.3 GB) from which every reported statistic is
computed are not in git. They will be archived with a DOI on publication and are available on request
until then.
