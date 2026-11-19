# Freeze record — PREREG_RTGCN_EXT.md (RT-GCN extension: NYSE_U, NASDAQ_T, NYSE_T)

**Frozen:** 2026-11-10 18:34 CET. Hashes computed on hyperion (/scratch/hilmibukhori/pcgv); code files
cross-checked identical to the laptop copy. NYSE data exist only on hyperion.

**Blinding statement:** at this time no extension confirm run has started, and no result of the main study's
RT-GCN NASDAQ confirm runs (`outputs/rtgcn_v1`) has been read by the analysis author (only file counts and
checkpoint epoch indices were inspected). The RSR stage-B results are likewise unread beyond the interim
descriptive look disclosed in the session (NASDAQ/NYSE placebo vs real, seed 0, partial K).

| file | sha256 |
|---|---|
| `PREREG_RTGCN_EXT.md` | `f787ea51a87dd552704307a56e2397ce22c2e5ccd7651a81f777e7e3deb83d43` |
| `configs/rtgcn_ext.yaml` | `23813de01e7f68430a912153344626410ca2abcbbcd575a52f265321b22c76b3` |
| `dapilot/rtgcn_ext.py` | `4f62f45d4f348b481383037b03b2a46317c19d76b76d39d4e7c6645c003c2b38` |
| `dapilot/rtgcn_ext_report.py` | `188c2f1a5d728d5819a949e7813373ade3d5ddc1d0934479704ff64f716544e6` |
| `scripts/run_rtgcn_ext.py` | `f471b80c01a138e4d7e170d9e86e2175d6ea711a6c7f35263fbe9648643b9d08` |
| `tests/test_rtgcn_ext.py` | `19bb40a12cfd410eb5238e862f44675ca6cec4ee35d22d2018691f9f23ec2d7c` |
| `scripts/hyperion/rtgcn_ext.sbatch` | `54c77f250fdcf2e6edd2f253f00d461c361deeb1a3cabda1d45d05c708ac70e9` |
| `scripts/hyperion/rtgcn_ext_tasks.txt` | `8feba283d4c3080dafca2ab8b8bf5203d7bf1acdff4322579edc604b9939d0d3` |
| `scripts/rtgcn_convert_labels.py` | `990f848f58f7bf7d09d08f2725a4e9177a8d53adb7560dba5b3cfb9f29568c47` |
| `dapilot/rtgcn_replica.py` | `d19420e1467d0d39fc03ba85fb8133cd3e98328bd8559018a8d2194a6f18dba1` |
| `dapilot/rtgcn_report.py` | `d018d5c73066ceaa289539b30fcccbeb874a97c0d0909e64f41f2b7b07e90052` |
| `data/raw/rtgcn_repo/net/rt_gcn.py` | `df5f19fd94a51bc89f153c7f75a1bdf696d0e701f7d31fbbcd178960ac6ee4e2` |
| `data/raw/rtgcn_repo/net/utils/graph.py` | `14e1553020552fd9c806af34c821bc025a3cf3f90872f44a93e95d4d06c33724` |
| `data/raw/rtgcn_repo/net/utils/tgcn.py` | `4cb508aa4d200d80b15c341507a6ceb1f5aae62e05109479f9f6a4a479aa2912` |
| `data/raw/rtgcn_repo/net/loss.py` | `dec9660d64f5deabaf361e5204d222d8b2a1a6f8023969f3e4295d608b6d14ff` |
| `data/raw/rtgcn_dl/NYSE/NYSE_relation.npy` | `1df6509173a3840e69a673fc98377fbbda047dbdd5f2506f04d8ecbf7f6d762e` |
| `data/raw/rtgcn_dl/NYSE/industry_labels_derived.npy` | `e87ce8f612b3f6294857978d4a0f16350aa77fb4c18d60f55b5fce94bbe4c6cc` |
| `data/raw/rtgcn_dl/NYSE/train_15_EOD.npy` | `bc130a9fb68e3e6bc4023d75fd9aa6c1b70db70aee2d60de1282041f239759fb` |
| `data/raw/rtgcn_dl/NYSE/test_15_EOD.npy` | `0fe01711054d523b12f8b5031ea74bad4c35f3d0c48897a61451059b049cff27` |
| `data/raw/rtgcn_dl/NYSE/train_15_price.npy` | `564100c3d4b83a0989429dd096b2133a69b665511cbea0cacb14b4b6bc167408` |
| `data/raw/rtgcn_dl/NYSE/test_15_price.npy` | `bb8b2d9b45b7beaec50ed9dd4e7f48f76dbf2d2d000b0293d5e2e92cccc34d0a` |
| `data/raw/rtgcn_dl/NYSE/train_15_label.npy` | `f68b0a7cb5b3484e212527c50ad38f50d6edf9aebaf7a6e9897a42ea14a33cf1` |
| `data/raw/rtgcn_dl/NYSE/test_15_label.npy` | `e5713682980a8b07e444e9b0558126cf4e9cb7ba04846a3c0c748942e05dcb00` |
| `data/raw/rtgcn_dl/NYSE/train_15_label.pkl` | `79da735c9c4ed93abe4ecf6790ff2b6ec1503aa0a6ba995b363f4b23622bc8dc` |
| `data/raw/rtgcn_dl/NYSE/test_15_label.pkl` | `a41de4efce388ff769f29832e7ae0a1640f29e68410c2512d791097a1c57b583` |
| `data/raw/rtgcn_dl/NYSE/NYSE_wiki_relation.npy` | `b6fbc179d0b66d3edb301e34063ff0c80a10735b496b2ac1b0c57d763ebf68d6` |
| `data/raw/rtgcn_dl/NYSE/NYSE_industry_relation.npy` | `aa6dc437ccd1cbc4219803422c6c65b2aa540c61332652b10ba57b410609f6af` |
| `data/raw/rtgcn_dl/NASDAQ/NASDAQ_relation.npy` | `3d56cadcb540138f15ecbe2e735693c41098c7e5617b6b0f731f4e6fd3a94e61` |
| `data/raw/rtgcn_dl/NASDAQ/industry_labels_derived.npy` | `145a5e22fa7c0bcacd9f543d1a731f1bc69d6f5bd5e01180aa242bdee6d77a5f` |
| `data/raw/rtgcn_dl/NASDAQ/train_15_EOD.npy` | `f4a85784821a319020f40abd1604c7490fbb3c712810b8466f67aaa1eec9b546` |
| `data/raw/rtgcn_dl/NASDAQ/test_15_EOD.npy` | `f781a85e62dc4f4e5d242e82dd1df25d7bcb089352e66d43646278d192197207` |
| `data/raw/rtgcn_dl/NASDAQ/train_15_price.npy` | `cbb07577a9def33fa62668d7ce3dadb2234df243760039b426252b443b7a8128` |
| `data/raw/rtgcn_dl/NASDAQ/test_15_price.npy` | `81997a02153c3549dcc156db1b9fcd273efe82461c2ef2f64ff80b555f48b2cd` |
| `data/raw/rtgcn_dl/NASDAQ/train_15_label.npy` | `9377b8c7865662d305f3cf9d3d1cfc45b31167a63961d0e75c911d0d0118ffa7` |
| `data/raw/rtgcn_dl/NASDAQ/test_15_label.npy` | `3eb308e450048235e71977dac9f53b19a6d39c4dffee83a04d55aed428de0d34` |
