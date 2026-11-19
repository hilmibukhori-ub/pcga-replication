import numpy as np
import torch

from dapilot import rtgcn_replica as rt


def test_patched_model_equals_original():
    """Same seed -> identical parameters, A and outputs as the model built by the ORIGINAL Graph from the file."""
    rel_path = str(rt.DATA / "NASDAQ" / "NASDAQ_relation.npy")
    torch.manual_seed(7)
    orig = rt.Model(in_channels=4, graph_args={"market": "NASDAQ", "relation_path": rel_path, "strategy": "uniform"},
                    edge_importance_weighting="Uniform", dropout=0.5)
    torch.manual_seed(7)
    ours = rt.build_model(rt.real_edge("NASDAQ"), rt.N_REL_CHANNELS["NASDAQ"])
    assert torch.equal(orig.A, ours.A)
    po, pn = dict(orig.named_parameters()), dict(ours.named_parameters())
    assert po.keys() == pn.keys()
    for k in po:
        assert torch.equal(po[k], pn[k]), k
    X, P, Y = rt.load("NASDAQ")["test"]
    x = torch.from_numpy(np.ascontiguousarray(X[:8][:, [0, 1, 2, 4]], dtype=np.float32))
    orig.eval(); ours.eval()
    with torch.no_grad():
        assert torch.allclose(orig(x), ours(x))


def test_loss_matches_original_formula():
    torch.manual_seed(0)
    p, g = torch.randn(4, 30), torch.randn(4, 30) * 0.02
    N, V = g.shape
    pp = p.view(N, V, 1) - p.view(N, 1, V)
    gg = g.view(N, 1, V) - g.view(N, V, 1)
    ref = torch.nn.functional.mse_loss(p, g) + 0.1 * torch.relu(pp * gg).mean()
    assert torch.allclose(rt.rt_loss(p, g, 0.1), ref)


def test_placebo_keeps_industry_and_moves_wiki():
    lab = rt.industry_labels("NASDAQ")
    pi = rt.within_label_perm(lab, np.random.default_rng(3))
    assert (lab[pi] == lab).all()
    ind = rt.real_edge("NASDAQ", channels=range(40, 136))
    assert np.array_equal(rt.permute_edge(ind, pi), ind)          # industry cliques invariant
    wiki = rt.real_edge("NASDAQ", channels=range(0, 40))
    assert not np.array_equal(rt.permute_edge(wiki, pi), wiki)    # wiki relations move


def test_daily_metrics_match_original_loops():
    rng = np.random.default_rng(1)
    res, lab = rng.normal(size=(5, 40)), rng.normal(size=(5, 40))
    d = rt.daily_metrics(res, lab)
    for i in range(5):                       # show_MRR / show_return_ration logic
        rank_gt, rank_pre = np.argsort(lab[i]), np.argsort(res[i])
        top1 = rank_pre[-1]; pos = 0
        for j in range(1, 41):
            pos += 1
            if rank_gt[-j] == top1:
                break
        assert np.isclose(d["rr"][i], 1 / pos)
        for k in (1, 5, 10):
            assert np.isclose(d[f"top{k}"][i], lab[i][rank_pre[-k:]].mean())


def test_checkpoint_resume_is_bit_identical(tmp_path):
    X, P, Y = rt.load("NASDAQ")["train"]
    tr, va, te = (X, P, Y, np.arange(0, 128)), (X, P, Y, np.arange(140, 204)), (X, P, Y, np.arange(210, 274))
    edge = rt.real_edge("NASDAQ")
    full = rt.run(edge, 136, 5, tr, va, te, epochs=3, lr=1e-3, wd=1e-4)
    ck = tmp_path / "c.pt"
    rt.run(edge, 136, 5, tr, va, te, epochs=2, lr=1e-3, wd=1e-4, ckpt=ck)      # "interrupted" after 2 epochs
    assert ck.exists()
    resumed = rt.run(edge, 136, 5, tr, va, te, epochs=3, lr=1e-3, wd=1e-4, ckpt=ck)
    assert full.keys() == resumed.keys()
    for k in full:
        assert np.array_equal(full[k], resumed[k]), k
