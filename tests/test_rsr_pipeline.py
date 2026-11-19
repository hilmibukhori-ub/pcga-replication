"""RSR pipeline tests: causality, target alignment, masks, test-lock, freeze guard, C1/C2, placebos.
Real-data tests are skipped when the RSR files are absent; none of them read 2017 values."""

import numpy as np
import pytest
import yaml

from dapilot import rsr_pipeline as rp
from dapilot.power import industry_edges, same_industry_counts
from dapilot.rsr_data import REL, industry_blocks, tickers, wiki_edges
from dapilot.stats import holm

from conftest import ROOT

HAVE_DATA = (REL / "wikidata/NASDAQ_wiki_relation.npy").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="RSR data not present")


def _panel(seed=0, T=80, n=30):
    rng = np.random.default_rng(seed)
    C = np.exp(np.cumsum(rng.normal(0, 0.02, (T, n)), 0)) * 10
    C[rng.random((T, n)) < 0.02] = np.nan
    return C


def test_features_are_causal():
    C = _panel()
    X = rp.features(C)
    C2 = C.copy()
    C2[50:] *= np.exp(np.random.default_rng(1).normal(0, 0.5, C2[50:].shape))
    X2 = rp.features(C2)
    np.testing.assert_allclose(X[:50], X2[:50], equal_nan=True)
    assert np.isnan(X[:20]).all() and np.isfinite(X[20:]).all()


def test_target_alignment_and_mask():
    C = _panel()
    y, m = rp.target(C)
    r = rp.returns(C)
    t, i = 30, int(np.flatnonzero(m[30])[0])
    valid = np.isfinite(r[t + 1])
    assert np.isclose(y[t, i], r[t + 1, i] - r[t + 1, valid].mean())
    np.testing.assert_array_equal(m[:-1], np.isfinite(C[:-1]) & np.isfinite(C[1:]))
    assert not m[-1].any()
    assert np.allclose(np.where(m, y, 0).sum(1)[m.any(1)], 0, atol=1e-12)   # cross-sectionally demeaned


def test_split_purge_and_dev_shift():
    dates = [f"2013-01-{d:02d}" for d in range(1, 31)]
    s = {"train": ["2013-01-01", "2013-01-10"], "val": ["2013-01-11", "2013-01-20"], "test": ["2013-01-21", "2013-01-30"]}
    r = rp.split_rows(dates, s)
    assert r == {"train": (0, 8), "val": (10, 18), "test": (20, 28)}
    cfg = yaml.safe_load((ROOT / "configs/rsr_prereg.yaml").read_text())
    d = rp.dev_split(cfg["data"]["split"])
    assert d["test"][1] < rp.TEST_START


def test_ridge_masked_rows_do_not_matter():
    rng = np.random.default_rng(0)
    T, n = 60, 20
    F = rng.normal(size=(T, n, 3))
    Y = F[..., 0] * 0.5 + rng.normal(size=(T, n))
    M = rng.random((T, n)) > 0.1
    rows = {"train": (0, 29), "val": (30, 44), "test": (45, 59)}
    a = rp.fit_ridge_losses(F, Y, Y, M, rows, [1.0, 10.0])
    Y2 = np.where(M, Y, 1e6)
    b = rp.fit_ridge_losses(F, Y2, Y2, M, rows, [1.0, 10.0])
    np.testing.assert_allclose(a["mse"], b["mse"])


def test_holm_and_rank_resolution():
    assert holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    p = rp.rank_test_p(np.array([0.0]), np.ones((99, 1)))
    assert p[0] == pytest.approx(0.01)


def test_freeze_guard_passes_and_detects_tamper(tmp_path, monkeypatch):
    rp.verify_freeze()
    fake = tmp_path / "PREREG_FREEZE.md"
    txt = (ROOT / "PREREG_FREEZE.md").read_text().replace("26d8479c", "00000000", 1)
    fake.write_text(txt)
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    (tmp_path / "PREREG_RSR.md").write_bytes((ROOT / "PREREG_RSR.md").read_bytes())
    with pytest.raises(rp.FreezeError):
        rp.verify_freeze()


@needs_data
def test_dev_load_excludes_test_rows():
    cfg = yaml.safe_load((ROOT / "configs/rsr_prereg.yaml").read_text())
    C, d = rp.load_close("NASDAQ", cfg["data"]["split"]["val"][1])
    assert d[-1] == "2016-12-30" and len(C) == 1008 and all(x < rp.TEST_START for x in d)


@needs_data
@pytest.mark.parametrize("m", ["NASDAQ", "NYSE"])
def test_c1_c2_and_l3_preservation(m):
    E = wiki_edges(m)
    assert (E[:, 0] < E[:, 1]).all()
    b = industry_blocks(m)
    assert len(b) == len(tickers(m))
    import json
    ind = json.loads((REL / f"sector_industry/{m}_industry_ticker.json").read_text())
    pos = {t: i for i, t in enumerate(tickers(m))}
    na = [pos[t] for t in ind.get("n/a", []) if t in pos]
    assert len(set(b[na])) == len(na)                            # C2: each n/a stock is a singleton
    E3 = industry_edges(E, b, np.random.default_rng(0))
    n = len(b)
    np.testing.assert_array_equal(np.bincount(E3.ravel(), minlength=n), np.bincount(E.ravel(), minlength=n))
    np.testing.assert_array_equal(same_industry_counts(E3, b, n), same_industry_counts(E, b, n))
