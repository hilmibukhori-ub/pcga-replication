# Freeze record — RT-GCN re-test pre-registration (v0.1 + pre-launch amendment RT1)

Frozen: 2026-11-10 14:53 CET, before any metric on the released 2020 test period was computed.
Supersedes `PREREG_RTGCN_FREEZE_v0.1_superseded.md` (14:47). The only change is robustness (DEVIATIONS RT1).
RT-GCN repository commit: `617c3f53a58753251929a0cdbd1c12f7e3df8154`.
Confirm modes verify every hash below at launch and refuse to run on a mismatch. Text and code files are read-only. Deviations: `DEVIATIONS.md`, prefix RT.

## Frozen files

| file | sha256 |
|---|---|
| `PREREG_RTGCN.md` | `d3fbcd73e7597e6850fe648702f045c5bbe04234020f0c8bcb34822477ceb567` |
| `configs/rtgcn_replica.yaml` | `639b1d78141db597ca878e4405d81a25d67eec9bef5ff9d35507bc5ceaa1077f` |
| `dapilot/rtgcn_replica.py` | `d19420e1467d0d39fc03ba85fb8133cd3e98328bd8559018a8d2194a6f18dba1` |
| `dapilot/rtgcn_report.py` | `d018d5c73066ceaa289539b30fcccbeb874a97c0d0909e64f41f2b7b07e90052` |
| `scripts/run_rtgcn_replica.py` | `9ddc932281de894238147b296af882b1005f114295ce5e5393ea541a099587bb` |
| `scripts/rtgcn_summarise.py` | `e0801dfbfcbf5981c88c2542beba8b1f3358d99283d7f3f0679e67273a6a4771` |
| `scripts/launch_rtgcn_after_rsr.sh` | `55738e2eea2fe9bb0907c19b2c4fda26dd5021d29554e967e8891b2c7a25745d` |
| `tests/test_rtgcn_replica.py` | `a4d5709bc141adb0ef38741d45dbf4aa285f071d0f146401dfd7eb87732013a8` |
| `data/raw/rtgcn_repo/net/rt_gcn.py` | `df5f19fd94a51bc89f153c7f75a1bdf696d0e701f7d31fbbcd178960ac6ee4e2` |
| `data/raw/rtgcn_repo/net/utils/graph.py` | `14e1553020552fd9c806af34c821bc025a3cf3f90872f44a93e95d4d06c33724` |
| `data/raw/rtgcn_repo/net/utils/tgcn.py` | `4cb508aa4d200d80b15c341507a6ceb1f5aae62e05109479f9f6a4a479aa2912` |
| `data/raw/rtgcn_repo/net/loss.py` | `dec9660d64f5deabaf361e5204d222d8b2a1a6f8023969f3e4295d608b6d14ff` |
| `data/raw/rtgcn_repo/processor/recognition.py` | `5afc7269b03481084c57746e697277e551ddf65870338bdf994304f0453b5b58` |
| `data/raw/rtgcn_dl/NASDAQ/industry_labels_derived.npy` | `145a5e22fa7c0bcacd9f543d1a731f1bc69d6f5bd5e01180aa242bdee6d77a5f` |
| `data/raw/rtgcn_dl/NASDAQ/NASDAQ_relation.npy` | `3d56cadcb540138f15ecbe2e735693c41098c7e5617b6b0f731f4e6fd3a94e61` |
| `data/raw/rtgcn_dl/NASDAQ/test_15_EOD.npy` | `f781a85e62dc4f4e5d242e82dd1df25d7bcb089352e66d43646278d192197207` |
| `data/raw/rtgcn_dl/NASDAQ/test_15_label.npy` | `3eb308e450048235e71977dac9f53b19a6d39c4dffee83a04d55aed428de0d34` |
| `data/raw/rtgcn_dl/NASDAQ/test_15_price.npy` | `81997a02153c3549dcc156db1b9fcd273efe82461c2ef2f64ff80b555f48b2cd` |
| `data/raw/rtgcn_dl/NASDAQ/train_15_EOD.npy` | `f4a85784821a319020f40abd1604c7490fbb3c712810b8466f67aaa1eec9b546` |
| `data/raw/rtgcn_dl/NASDAQ/train_15_label.npy` | `9377b8c7865662d305f3cf9d3d1cfc45b31167a63961d0e75c911d0d0118ffa7` |
| `data/raw/rtgcn_dl/NASDAQ/train_15_price.npy` | `cbb07577a9def33fa62668d7ce3dadb2234df243760039b426252b443b7a8128` |
