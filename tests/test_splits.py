import numpy as np
import pytest

from dapilot.splits import SPLIT_ORDER, LeakageError, TemporalSplits


def test_splits_cover_and_order(cfg):
    sp = TemporalSplits(cfg["splits"], cfg["data"]["n_steps"])
    prev = -1
    for n in SPLIT_ORDER:
        assert sp.start(n) == prev + 1
        prev = sp.end(n)
    assert prev == cfg["data"]["n_steps"] - 1
    assert sp.start("test") == 2000 and sp.end("expert_train") == 799   # 1-based 2001 / 800


def test_purged_decision_times_have_labels_inside_split(cfg):
    sp = TemporalSplits(cfg["splits"], cfg["data"]["n_steps"])
    for n in SPLIT_ORDER:
        d = sp.decision_times(n)
        assert d.min() == sp.start(n)
        assert d.max() + sp.horizon == sp.end(n)
        sp.assert_labels_within(d, n)


def test_guards_raise(cfg):
    sp = TemporalSplits(cfg["splits"], cfg["data"]["n_steps"])
    with pytest.raises(LeakageError):
        sp.assert_labels_within(np.array([sp.end("expert_train")]), "expert_train")
    with pytest.raises(LeakageError):
        sp.assert_labels_before(np.array([sp.start("test") - 1]), "test")
    sp.assert_labels_before(sp.decision_times("calibration"), "test")


def test_non_contiguous_rejected(cfg):
    bad = dict(cfg["splits"])
    bad["expert_val"] = [802, 1000]
    with pytest.raises(ValueError):
        TemporalSplits(bad, cfg["data"]["n_steps"])
