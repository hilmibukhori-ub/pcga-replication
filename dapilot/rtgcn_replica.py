"""Re-test harness for RT-GCN (Zheng, Shao, Zhu & Shen, ICDE 2023), released 'Uniform' variant, on the
authors' released NASDAQ data (data/raw/rtgcn_dl). The ORIGINAL model code (data/raw/rtgcn_repo, commit
617c3f53) is imported unchanged. Only these things are patched:

- `torch.Tensor.cuda` becomes a no-op (CPU run). The Uniform forward path does not call .cuda().
- `Graph.get_edge` is replaced by a version that takes a precomputed edge matrix (eye + any-relation, exactly
  as the original builds it) and a tiny (1, 1, n_channels) relation placeholder. The Uniform forward path
  never reads `relation`. Its channel count only sizes an unused nn.Linear, so the count is kept identical
  (same RNG consumption, so the same parameter initialisation). A test checks that the parameters and
  outputs equal those of the model built by the original Graph from the full relation file.

The loss is re-implemented 1:1 from net/loss.py, which hard-codes device='cuda'. Training follows
processor/recognition.py:
- Adam (default optimizer) with base_lr and weight_decay from config/rt_gcn/NASDAQ/train.yaml
  (0.1 / 0.1) and no LR schedule (`step` applies to SGD only).
- batch_size 64, shuffle=True, drop_last=True; test loader batch 64, shuffle=False.
- input channels [0, 1, 2, 4]; prediction = (out - price) / price; loss alpha = 0.1; num_epoch = 100.
- "Published" epoch selection is the minimum TEST mean loss (mean of per-batch losses, day 89
  included). The published metrics delete test day 89.

Every epoch is evaluated on validation and test, so each selection rule can be applied afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/rtgcn_repo"
DATA = ROOT / "data/raw/rtgcn_dl"
N_REL_CHANNELS = {"NASDAQ": 136}          # non-self channels (raw tensor has one more: identity)
WIKI_CHANNELS = {"NASDAQ": range(0, 40)}  # channels 40..135 are industry cliques (verified)
DELETE_DAY = 89

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
torch.Tensor.cuda = lambda self, *a, **k: self          # CPU
from net.rt_gcn import Model                              # noqa: E402
from net.utils import graph as _graph_mod                 # noqa: E402

_CURRENT = {}


def _get_edge(self, market, relation_path):
    """Replacement for Graph.get_edge: uses _CURRENT['edge'] (V x V) and a tiny relation placeholder."""
    edge = _CURRENT["edge"]
    self.num_node = edge.shape[0]
    self.relation = np.zeros((1, 1, _CURRENT["n_ch"]))
    self.edge = edge


def build_model(edge, n_ch, mode="Uniform"):
    orig = _graph_mod.Graph.get_edge
    _CURRENT.update(edge=edge, n_ch=n_ch)
    _graph_mod.Graph.get_edge = _get_edge
    try:
        m = Model(in_channels=4, graph_args={"market": "NASDAQ", "relation_path": None, "strategy": "uniform"},
                  edge_importance_weighting=mode, dropout=0.5)
    finally:
        _graph_mod.Graph.get_edge = orig
    return m


# ----------------------------------------------------------------------------- data and graphs
def load(market="NASDAQ"):
    d = DATA / market
    out = {}
    for part in ("train", "test"):
        out[part] = (np.load(d / f"{part}_15_EOD.npy", mmap_mode="r"),
                     np.load(d / f"{part}_15_price.npy", allow_pickle=False),
                     np.load(d / f"{part}_15_label.npy", allow_pickle=False))
    return out


def real_edge(market="NASDAQ", channels=None):
    """eye + (any non-self relation > 0), exactly as Graph.get_edge (diagonal can be 2 for industry members)."""
    r = np.load(DATA / market / f"{market}_relation.npy", mmap_mode="r")
    rel = np.asarray(r[:, :, :-1]) if channels is None else np.asarray(r[:, :, list(channels)])
    any_ = (rel.sum(2) >= 1).astype(np.float64)
    return np.eye(r.shape[0]) + any_


def industry_labels(market="NASDAQ"):
    """Derived from the diagonal membership of industry channels (verified: edges == same-label pairs).
    No-industry stocks get label -1 and are pooled into one permutation group."""
    return np.load(DATA / market / "industry_labels_derived.npy", allow_pickle=False)


def within_label_perm(labels, rng):
    pi = np.arange(len(labels))
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        pi[idx] = idx[rng.permutation(len(idx))]
    return pi


def permute_edge(edge, pi):
    return edge[np.ix_(pi, pi)]


# ----------------------------------------------------------------------------- loss / metrics
def rt_loss(prediction, gt, alpha=0.1):
    """1:1 with net/loss.py (minus device='cuda')."""
    N, V = gt.size()
    p, g = prediction.view(N, V, 1), gt.view(N, V, 1)
    one = torch.ones((V, 1), dtype=torch.float32)
    pp = torch.sub(torch.matmul(p, one.permute(1, 0)), torch.matmul(one, p.permute(0, 2, 1)))
    gg = torch.sub(torch.matmul(one, g.permute(0, 2, 1)), torch.matmul(g, one.permute(1, 0)))
    return torch.nn.functional.mse_loss(p, g) + alpha * torch.mean(torch.relu(pp * gg))


def daily_metrics(result, label):
    """Per day: top-1/5/10 mean realised return (show_return_ration) and reciprocal rank (show_MRR)."""
    D = result.shape[0]
    out = {k: np.zeros(D) for k in ("top1", "top5", "top10", "rr")}
    for i in range(D):
        order = np.argsort(result[i])[::-1]
        for k in (1, 5, 10):
            out[f"top{k}"][i] = label[i][order[:k]].mean()
        gt_order = np.argsort(label[i])[::-1]
        out["rr"][i] = 1.0 / (int(np.where(gt_order == order[0])[0][0]) + 1)
    return out


# ----------------------------------------------------------------------------- training
def _batches(X, P, Y, idx, bs):
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        yield (torch.from_numpy(np.ascontiguousarray(X[j][:, [0, 1, 2, 4]], dtype=np.float32)),
               torch.from_numpy(np.ascontiguousarray(P[j], dtype=np.float32)),
               torch.from_numpy(np.ascontiguousarray(Y[j], dtype=np.float32)))


def evaluate(model, X, P, Y, idx, bs=64):
    """Mean of per-batch losses (as recognition.test) and predictions for idx (in order)."""
    model.eval()
    losses, preds = [], []
    with torch.no_grad():
        for x, cp, y in _batches(X, P, Y, idx, bs):
            pr = (model(x) - cp) / cp
            losses.append(float(rt_loss(pr, y, 0.1)))
            preds.append(pr.numpy())
    return float(np.mean(losses)), np.concatenate(preds)


def splits(mode, n_train=1295, n_val=252):
    """Index sets into the released TRAIN arrays (and the TEST arrays for 'test').
    confirm/pub : train = all 1295, no val, test = released test (published protocol).
    confirm/val : train = [0, n-252-1), val = [n-252, n), test = released test (one sample purged).
    dev         : train = [0, n-504-1), val = [n-504, n-252-1), pseudo-test = [n-252, n) of TRAIN.
    devpub      : train = [0, n-252-1), no val, pseudo-test = [n-252, n) of TRAIN.
                  The released test arrays are never loaded in dev."""
    n = n_train
    if mode == "pub":
        return np.arange(n), np.arange(0), "test"
    if mode == "val":
        return np.arange(n - n_val - 1), np.arange(n - n_val, n), "test"
    if mode == "dev":
        return np.arange(n - 2 * n_val - 1), np.arange(n - 2 * n_val, n - n_val - 1), np.arange(n - n_val, n)
    if mode == "devpub":   # dev analogue of 'pub': train on everything before the pseudo-test, no val
        return np.arange(n - n_val - 1), np.arange(0), np.arange(n - n_val, n)
    raise ValueError(mode)


def _save_ckpt(path, ep, model, opt, gen, hist, daily):
    """Atomic per-epoch checkpoint: everything needed for a bit-identical continuation."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"ep": ep, "model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                "hist": hist, "daily": daily}, tmp)
    tmp.replace(path)


def run(edge, n_ch, seed, train, val, test, epochs=100, lr=0.1, wd=0.1, bs=64, log=None, ckpt=None):
    """train/val/test: (X, P, Y, idx) tuples (val idx may be empty).
    ckpt: optional path. State is saved after every epoch, and an existing checkpoint is resumed from.
    The continuation is bit-identical to an uninterrupted run (tested)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(edge, n_ch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xtr, Ptr, Ytr, train_idx = train
    gen = torch.Generator().manual_seed(seed)
    hist = {k: [] for k in ("train_loss", "val_loss", "test_loss")}
    daily = {f"{part}_{k}": [] for part in ("val", "test") for k in ("top1", "top5", "top10", "rr")}
    start = 0
    if ckpt is not None and Path(ckpt).exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); gen.set_state(st["gen"])
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["np_rng"])
        hist, daily, start = st["hist"], st["daily"], st["ep"] + 1
        if log:
            log(f"  resumed from checkpoint after epoch {st['ep']}")
    for ep in range(start, epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        n_full = (len(order) // bs) * bs                      # drop_last=True
        tl = []
        for x, cp, y in _batches(Xtr, Ptr, Ytr, order[:n_full], bs):
            loss = rt_loss((model(x) - cp) / cp, y, 0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
        hist["train_loss"].append(float(np.mean(tl)))
        for part, (X, P, Y, idx) in (("val", val), ("test", test)):
            if len(idx) == 0:
                continue
            l, pr = evaluate(model, X, P, Y, idx, bs)
            hist[f"{part}_loss"].append(l)
            for k, v in daily_metrics(pr, Y[idx]).items():
                daily[f"{part}_{k}"].append(v)
        if ckpt is not None:
            _save_ckpt(ckpt, ep, model, opt, gen, hist, daily)
        if log:
            vl = hist["val_loss"][-1] if hist["val_loss"] else float("nan")
            log(f"  ep {ep} train {hist['train_loss'][-1]:.4g} val {vl:.4g} test {hist['test_loss'][-1]:.4g} "
                f"test top1-sum {daily['test_top1'][-1].sum():+.3f}")
    return {**{k: np.array(v) for k, v in hist.items()}, **{k: np.array(v) for k, v in daily.items() if v}}
