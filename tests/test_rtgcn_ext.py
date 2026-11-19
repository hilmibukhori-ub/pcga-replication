"""Tests for the RT-GCN extension harness (dapilot/rtgcn_ext.py). Heavy (full relation tensors): run on hyperion."""
import numpy as np
import pytest
import torch

from dapilot import rtgcn_replica as rt
from dapilot import rtgcn_ext as ext
from net.rt_gcn import Model


def _orig(market, mode):
    """Model built by the ORIGINAL Graph from the released relation file."""
    return Model(in_channels=4, graph_args={"market": market, "relation_path": str(rt.DATA / market / f"{market}_relation.npy"),
                                            "strategy": "uniform"}, edge_importance_weighting=mode, dropout=0.5)


@pytest.mark.parametrize("market,variant", [("NYSE", "U"), ("NASDAQ", "T"), ("NYSE", "T")])
def test_patched_equals_original(market, variant):
    mode = {"U": "Uniform", "T": "Time-aware"}[variant]
    torch.manual_seed(0); m0 = _orig(market, mode)
    edge, rel = ext.graph_for(market, "real")
    torch.manual_seed(0); m1 = ext.build_model_ext(edge, rel, variant)
    s0, s1 = m0.state_dict(), m1.state_dict()
    for k in s0:
        if k == "relation" and variant == "U":
            continue                                    # U never reads it (placeholder)
        assert torch.equal(s0[k].float(), s1[k].float()), k
    X = np.load(rt.DATA / market / "train_15_EOD.npy", mmap_mode="r")
    x = torch.from_numpy(np.ascontiguousarray(X[:4][:, [0, 1, 2, 4]], dtype=np.float32))
    m0.eval(); m1.eval()
    with torch.no_grad():
        assert torch.allclose(m0(x), m1(x), atol=0, rtol=0)


@pytest.mark.parametrize("market", ["NASDAQ", "NYSE"])
def test_channel_layout_and_labels(market):
    R = ext.relation(market)
    assert R.shape == (ext.N_NODES[market], ext.N_NODES[market], ext.N_REL[market])
    lab = ext.industry_labels(market)
    ind = R[:, :, ext.WIKI[market].stop:]
    same = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    assert np.array_equal(ind.sum(2) > 0, same)          # industry channels are exactly same-label cliques


@pytest.mark.parametrize("market", ["NASDAQ", "NYSE"])
def test_perm_moves_only_wiki(market):
    _, R = ext.graph_for(market, "real")
    _, P = ext.graph_for(market, "perm", k=0)
    w = ext.WIKI[market]
    assert np.array_equal(R[:, :, w.stop:], P[:, :, w.stop:])      # industry invariant
    assert not np.array_equal(R[:, :, list(w)], P[:, :, list(w)])  # wiki moved
    assert R[:, :, list(w)].sum() == P[:, :, list(w)].sum()        # same number of wiki edges


def test_none_has_no_relation():
    e, R0 = ext.graph_for("NYSE", "none")
    assert np.array_equal(e, np.eye(1405)) and not R0.any()
