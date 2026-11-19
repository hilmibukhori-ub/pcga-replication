"""Tests for Revision 1 (PREREG_REV1.md, dev rehearsal item 1). Return-free."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dapilot import rev1_sizebin as sb
from dapilot import rsr_replica as rr
from dapilot.rev1_report import edge_shift80, rank_test

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/rev1.yaml").read_text())
HAVE_RSR = (rr.REPO / "data/google_finance").exists()
HAVE_DATES = (rr.REPO / "data/NASDAQ_aver_line_dates.csv").exists()


@pytest.mark.skipif(not HAVE_DATES, reason="RSR calendar not on this machine")
def test_training_window_is_processed_rows_0_to_755():
    for m in ("NASDAQ", "NYSE"):
        w = sb.training_dates(m)
        assert len(w) == rr.VALID_INDEX == 756
        assert w[0] == "2013-01-02 16:00:00"
        d = np.genfromtxt(rr.REPO / f"data/{m}_aver_line_dates.csv", dtype=str, delimiter=",")
        first_val = str(d[29 + rr.VALID_INDEX])[:19]
        assert max(w) < first_val                      # no validation or test date in the window


def test_size_ignores_dates_outside_window_and_zero_volume():
    window = [f"2014-01-{d:02d} 16:00:00" for d in range(1, 29)]
    idx = [w + "-05:00" for w in window]
    df = pd.DataFrame({"Close": np.full(28, 10.0), "Volume": np.full(28, 100)}, index=idx)
    base = sb.size_from_frame(df, window, min_days=5)
    future = pd.DataFrame({"Close": [1e6] * 3, "Volume": [1e9] * 3},
                          index=["2016-06-01 16:00:00-04:00", "2017-03-01 16:00:00-05:00", "2017-06-01 16:00:00-04:00"])
    assert sb.size_from_frame(pd.concat([df, future]), window, min_days=5) == base == 1000.0
    df2 = df.copy()
    df2.iloc[:20, df2.columns.get_loc("Volume")] = 0
    assert sb.size_from_frame(df2, window, min_days=10) == -np.inf     # 8 valid days < 10


def test_sizebin_labels_split_each_group_lower_floor_half():
    parent = np.array([0, 0, 0, 0, 0, 1, 1, 2])
    size = np.array([5., 1., 3., 4., 2., 9., 7., 1.])
    lab = sb.sizebin_labels(parent, size)
    assert (lab // 2 == parent).all()
    # group 0 (n=5): lower floor(5/2)=2 by size are indices 1 (1.) and 4 (2.)
    assert set(np.where(lab == 0)[0]) == {1, 4} and set(np.where(lab == 1)[0]) == {0, 2, 3}
    assert lab[6] == 2 and lab[5] == 3                  # group 1: smaller -> bin 0
    assert lab[7] == 5                                  # singleton -> bin 1, fixed by permutation


def test_ties_broken_by_index():
    lab = sb.sizebin_labels(np.zeros(4, int), np.ones(4))
    assert lab.tolist() == [0, 0, 1, 1]


def test_rng_stream_distinct_from_parent():
    lab = np.zeros(200, int)
    assert not np.array_equal(sb.sizebin_pi(lab, 20261110, 0, 0), sb.parent_pi(lab, 20261110, 0, 0))
    assert np.array_equal(sb.parent_pi(lab, 20261110, 0, 3),
                          rr.within_label_perm(lab, np.random.default_rng([20261110, 0, 7, 3])))


@pytest.mark.skipif(not HAVE_RSR, reason="RSR raw data not on this machine")
@pytest.mark.parametrize("market,mi", [("NASDAQ", 0), ("NYSE", 1)])
def test_real_data_permutation_preserves_industry_bin_and_edges(market, mi):
    size = sb.size_measure(market)
    assert np.isfinite(size).all()
    parent = rr.industry_labels(market)
    lab = sb.sizebin_labels(parent, size)
    rel = rr.load_relation(market, "wikidata")
    E = rr.to_edges(rel)
    for k in (0, 1, 98):
        pi = sb.sizebin_pi(lab, 20261110, mi, k)
        assert (lab[pi] == lab).all() and (parent[pi] == parent).all()
        assert sorted(pi.tolist()) == list(range(len(pi)))
        P = rr.permute_edges(E, pi)
        assert len(P[0]) == len(E[0]) and ((P[0] != P[1]).sum() == (E[0] != E[1]).sum())
        assert np.array_equal(P[2], E[2])               # typed relation vectors move with the edge
        # dense check: R'[i,j] = R[pi[i], pi[j]]
        a, b = P[0][0], P[1][0]
        assert np.array_equal(rel[pi[a], pi[b]], P[2][0])


def test_k_task_list_exact():
    lines = (ROOT / "scripts/hyperion/rev1_k99_tasks.txt").read_text().strip().splitlines()
    assert len(lines) == 160
    want = {"NASDAQ_U": "scripts/run_rtgcn_replica.py --mode val --cfg paper --graph perm --k {k} --seeds 0",
            "NYSE_U": "scripts/run_rtgcn_ext.py --market NYSE --variant U --mode val --graph perm --k {k} --seeds 0"}
    for i, cell in enumerate(("NASDAQ_U", "NYSE_U")):
        block = lines[80 * i:80 * (i + 1)]
        assert block == [f"{cell} " + want[cell].format(k=k) for k in range(19, 99)]
    assert CFG["study_K"]["new_k"] == [19, 99] and CFG["study_K"]["K"] == 99


def test_frozen_runners_and_reports_unchanged():
    """Study K must call the parent code exactly as frozen."""
    for freeze in ("PREREG_RTGCN_FREEZE.md", "PREREG_RTGCN_EXT_FREEZE.md", "PREREG_REPLICATION_FREEZE.md"):
        if not (ROOT / freeze).exists():
            continue                                    # e.g. the RSR freeze record is laptop-only
        rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", (ROOT / freeze).read_text())
        for p, h in rows:
            if (ROOT / p).exists() and (p.endswith(".py") or p.endswith(".yaml")):
                assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p


def test_rank_test_and_power():
    r = rank_test(1.0, [0.0] * 99)
    assert r["p_rank"] == 0.01 and r["rank_of_real"] == 1
    r = rank_test(0.0, [0.0] * 19)                       # ties count against the real graph
    assert r["p_rank"] == 1.0
    s99, s19 = edge_shift80(99, 0.05, reps=20000), edge_shift80(19, 0.05, reps=20000)
    assert 2.2 < s99 < 2.8 and s99 < s19 < 3.2          # paper §power: about 2.5 (K=99) and 2.8 (K=19)
