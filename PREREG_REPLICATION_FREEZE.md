# Freeze record — RSR replication pre-registration

Frozen: 2026-11-10 10:54 CET, before any RSR-model output on the 2017 test period was computed.
Confirm mode (`scripts/run_rsr_replica.py --mode confirm`) verifies every hash below at launch and refuses to run on a mismatch.
Frozen text and code files are set read-only. Deviations go to `DEVIATIONS.md` with prefix RP.
RSR repository commit: `cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d`. Embedding archive sha256: `205973c74fb15fa46f10c96f92280aa38bcee1d8067357ced3500223b846acd0`.

## Frozen files

| file | sha256 |
|---|---|
| `PREREG_RSR_REPLICATION.md` | `0d542f9d710263281d03b46843d8b036f3d1053988a8e5263596c5c2c24eda83` |
| `configs/rsr_replica.yaml` | `fa213898e6ef8b8b10b903d688655fe9ea020f942d79b996eacad5c06e0083a6` |
| `dapilot/rsr_replica.py` | `cfe792cc228ab79b35233d96ce241c728565269387086a529d886bd46f142536` |
| `dapilot/rsr_replica_infer.py` | `73795c003846ebf25a863742a9a9b94dffe0cdabc179d0f48410490ce7d69f91` |
| `dapilot/rsr_replica_report.py` | `efc92dabc5c7961ed724a78cb2ba7765a81a40002f9ed0cdf6246b224fc64367` |
| `dapilot/rsr_pipeline.py` | `812f5d8ad1b880c35f25bb5f4e1a74d869c5701ad318e607ac241d7bccf63e73` |
| `dapilot/stats.py` | `f9c7ed38ebea22949b9799ebbef7f98969b292b99e65d616fdc603bb8d21eb0c` |
| `dapilot/power.py` | `f434f0312eb6992b8223cf8b66b5c9a4889d49d6eff0ca84fa9c82e29de48fa7` |
| `dapilot/power_v2.py` | `382dfcf4e0b51999bacf93d364e13986c814980e598b1a0bf9809771e2c25dfe` |
| `scripts/run_rsr_replica.py` | `b23ad7ce770fbcc00525ff4f47e5dd4b872e89bd66639eba85cd455d7d87f29b` |
| `tests/test_rsr_replica.py` | `a7b7c1b5c7baa4b2fbd5b20a4ddad99dc8bf4b045b7243c6a009e8f43b43f2f4` |
| `literature/feng2019_rsr_1809.09441.pdf` | `735675e846e548f35c139ad96c6fda2f6a936a3f65150b0a249ad36fffcc386f` |
| `data/raw/rsr_pretrain_dl/pretrain/NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy` | `071a13ade269a50a5aa77bab152fe408b5ebb0ea2ff79aa73268c1720c4301b2` |
| `data/raw/rsr_pretrain_dl/pretrain/NYSE_rank_lstm_seq-8_unit-32_0.csv.npy` | `c848999b2085d6fdabed1c30eb92454f929a54a1506b0e5a3718ad3d5917f6d8` |
| `data/raw/rsr_relation/relation/wikidata/NASDAQ_wiki_relation.npy` | `00b0df3a3fb387f451aa0466fa475651ad6d9b244c1bb9fa43209c28b5f24d40` |
| `data/raw/rsr_relation/relation/wikidata/NYSE_wiki_relation.npy` | `42416de7aa3e2eb6d3ba546473d855344503ff9b34951f7269d579bcefd69c22` |
| `data/raw/rsr_relation/relation/sector_industry/NASDAQ_industry_relation.npy` | `4838497b16fd6d900c4c93b12c6571cc24909ad3123377be062417a03d12e3fd` |
| `data/raw/rsr_relation/relation/sector_industry/NYSE_industry_relation.npy` | `41ab68f29541e11e2acf0ec7df9ca68f9c22d8ebcca3bba64a8f128454d56883` |
| `data/raw/rsr_relation/relation/sector_industry/NASDAQ_industry_ticker.json` | `1e3fa58416d0f16783368fc1a7cc4a118e8067c1a4ee8667a1ae46a40747e852` |
| `data/raw/rsr_relation/relation/sector_industry/NYSE_industry_ticker.json` | `8cc29ef64be3fec5bf2264dc930ed20dc105e1ec3273e0f556ab64a90ab44f7f` |
