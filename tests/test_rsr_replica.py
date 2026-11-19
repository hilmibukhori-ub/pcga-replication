import numpy as np
import torch

from dapilot import rsr_replica as rr


def _rand_rel(n, R, p, rng):
    rel = (rng.random((n, n, R)) < p).astype(np.int64)
    rel[:, :, -1] = np.eye(n, dtype=np.int64)
    return rel


def test_sparse_matches_dense_formula():
    rng = np.random.default_rng(0)
    for n, R, p in [(7, 3, 0.2), (40, 5, 0.03), (25, 2, 0.0)]:
        rel = _rand_rel(n, R, p, rng)
        torch.manual_seed(1)
        model = rr.RSR(8, R)
        f = torch.randn(n, 8)
        i, j, r = (torch.from_numpy(x) for x in rr.to_edges(rel))
        with torch.no_grad():
            sp = model(f, i.long(), j.long(), r)
        assert torch.allclose(sp, rr.dense_reference(model, f, rel), atol=1e-5)


def test_sparse_matches_dense_on_real_wiki():
    rel = rr.load_relation("NASDAQ", "wikidata")
    torch.manual_seed(2)
    model = rr.RSR(16, rel.shape[2])
    f = torch.randn(rel.shape[0], 16)
    i, j, r = (torch.from_numpy(x) for x in rr.to_edges(rel))
    with torch.no_grad():
        sp = model(f, i.long(), j.long(), r)
    assert torch.allclose(sp, rr.dense_reference(model, f, rel), atol=1e-5)


def test_self_only_is_no_relation_baseline():
    n, R = 30, 4
    torch.manual_seed(3)
    model = rr.RSR(8, R)
    f = torch.randn(n, 8)
    i, j, r = (torch.from_numpy(x) for x in rr.self_only_edges(n, R))
    with torch.no_grad():
        out = model.propagate(f, i.long(), j.long(), r)
    assert torch.allclose(out, f)


def test_within_label_perm_placebo():
    rng = np.random.default_rng(4)
    rel = _rand_rel(50, 4, 0.05, rng)
    lab = rng.integers(0, 5, 50)
    pi = rr.within_label_perm(lab, rng)
    assert (lab[pi] == lab).all()
    i, j, r = rr.permute_edges(rr.to_edges(rel), pi)
    dense = np.zeros_like(rel)
    dense[i, j] = r.astype(np.int64)
    assert (dense == rel[pi][:, pi]).all()                       # R'[a,b] = R[pi a, pi b]
    assert (dense[:, :, -1] == np.eye(50)).all()                 # self channel preserved
    # industry-pair composition of edges preserved, per relation type
    for k in range(rel.shape[2]):
        a, b = np.nonzero(rel[:, :, k]); a2, b2 = np.nonzero(dense[:, :, k])
        assert sorted(zip(lab[a], lab[b])) == sorted(zip(lab[a2], lab[b2]))


def test_load_eod_semantics():
    mask, gt, base = rr.load_eod("NYSE")
    assert mask.shape == (1737, 1245)
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert (base[mask == 0] == np.float32(1.1)).all()
    assert (gt[:, 0] == 0).all()


def test_rank_loss_equals_tf_formula():
    torch.manual_seed(5)
    N = 300
    pred = torch.rand(N, 1, requires_grad=True) + 0.5
    base = torch.rand(N, 1) + 0.5
    gt = torch.randn(N, 1) * 0.02
    m = (torch.rand(N, 1) > 0.1).float()
    tot, reg, rank, r = rr.losses(pred, base, gt, m, 3.0)
    ref_rank = torch.relu((r - r.T) * (gt.T - gt) * (m @ m.T)).mean()
    ref_reg = (m * (r - gt) ** 2).sum() / (m != 0).sum()
    assert torch.allclose(rank, ref_rank, rtol=1e-5) and torch.allclose(reg, ref_reg)
    g1 = torch.autograd.grad(tot, pred, retain_graph=True)[0]
    g2 = torch.autograd.grad(ref_reg + 3.0 * ref_rank, pred)[0]
    assert torch.allclose(g1, g2, atol=1e-6)
