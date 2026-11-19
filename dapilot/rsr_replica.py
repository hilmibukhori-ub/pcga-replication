"""Faithful PyTorch port of RSR (Feng et al. 2019, TOIS) `training/relation_rank_lstm.py`, default
"sum weight" TGC = RSR_I, on top of the authors' frozen pretrained Rank_LSTM embeddings.

Fidelity notes (all mirror the TF code in data/raw/rsr_repo/training):
- data: load_EOD_data semantics (mask = close observed; gt = 1-step return when both closes observed;
  base price = close with the -1234 missing code replaced by 1.1); NASDAQ drops its last row.
- batch = all stocks on one day; offsets 0..valid_index-1 shuffled each epoch, the first
  (valid_index - seq - steps + 1) used (so training labels reach seq days into validation, as in
  the original); mask = min over [offset, offset + seq + steps).
- relational layer: rel_weight = leaky(dense(r_ij)), weight = head_i + tail_j + rel_ij, softmax over
  i for every column j (TF dim=0) restricted to pairs with any relation (the -1e9 mask), including
  the self-relation channel, then out = W @ F. Implemented sparsely; tests check it against the
  dense formula.
- prediction = leaky(dense([f, out])); return_ratio = (pred - base)/base; loss = weighted MSE
  (SUM_BY_NONZERO_WEIGHTS) + alpha * mean relu pairwise rank loss. Adam lr 1e-3, 50 epochs.
- selection: epoch with the lowest validation total loss; the test metrics of that epoch.
- dense layers: glorot-uniform kernels, zero biases (TF1 tf.layers.dense defaults); leaky alpha 0.2.

Only the relational layer and the output layer are trained; the embeddings are fixed (as in the
original). The no-relation baseline (matched Rank_LSTM head) is the same model with only the
self-relation channel, so out = f.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/rsr_repo"
REL = ROOT / "data/raw/rsr_relation/relation"
EMB = ROOT / "data/raw/rsr_pretrain_dl/pretrain"
VALID_INDEX, TEST_INDEX = 756, 1008
LEAK = 0.2


# ----------------------------------------------------------------------------- data
def tickers(market):
    return np.genfromtxt(REPO / f"data/{market}_tickers_qualify_dr-0.98_min-5_smooth.csv", dtype=str).tolist()


def load_eod(market, steps=1):
    """mask, gt, base_price: (N, T) exactly as load_EOD_data."""
    masks, gts, base = [], [], []
    for t in tickers(market):
        a = np.genfromtxt(REPO / f"data/2013-01-01/{market}_{t}_1.csv", dtype=np.float32, delimiter=",")
        if market == "NASDAQ":
            a = a[:-1]
        c = a[:, -1]
        miss = np.abs(c + 1234) < 1e-8
        m = (~miss).astype(np.float32)
        g = np.zeros_like(c)
        prev_ok = np.zeros_like(miss)
        prev_ok[steps:] = ~miss[:-steps]
        ok = (~miss) & prev_ok
        g[steps:][ok[steps:]] = ((c[steps:] - c[:-steps]) / c[:-steps])[ok[steps:]]
        masks.append(m)
        gts.append(g)
        base.append(np.where(miss, np.float32(1.1), c))
    return np.stack(masks), np.stack(gts), np.stack(base)


def load_embedding(fname):
    return np.load(EMB / fname, allow_pickle=False, mmap_mode="r")


def load_relation(market, name):
    tail = {"wikidata": "_wiki_relation.npy", "sector_industry": "_industry_relation.npy"}[name]
    return np.load(REL / f"{name}/{market}{tail}", allow_pickle=False)


def to_edges(rel):
    """(N,N,R) encoding -> (row i, col j, R-vector) for every pair with any relation (incl. self)."""
    i, j = np.nonzero(rel.sum(2))
    return i, j, rel[i, j].astype(np.float32)


def self_only_edges(N, R):
    r = np.zeros((N, R), np.float32)
    r[:, -1] = 1.0
    ii = np.arange(N)
    return ii, ii, r


def industry_labels(market):
    """Industry label per stock with `n/a` pooled into ONE label (used only for permutation placebos)."""
    tick = tickers(market)
    pos = {t: k for k, t in enumerate(tick)}
    ind = json.loads((REL / f"sector_industry/{market}_industry_ticker.json").read_text())
    lab = np.full(len(tick), -1)
    for g_i, g in enumerate(sorted(ind)):
        for t in ind[g]:
            if t in pos:
                lab[pos[t]] = g_i
    nxt = lab.max() + 1
    for k in np.where(lab < 0)[0]:          # stocks absent from the json: own label (fixed)
        lab[k] = nxt
        nxt += 1
    return lab


def within_label_perm(labels, rng):
    """pi with labels[pi[i]] == labels[i]."""
    pi = np.arange(len(labels))
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        pi[idx] = idx[rng.permutation(len(idx))]
    return pi


def permute_edges(edges, pi):
    """Placebo R'[i,j] = R[pi[i], pi[j]]: an edge (a,b) moves to (pi^-1(a), pi^-1(b))."""
    inv = np.empty_like(pi)
    inv[pi] = np.arange(len(pi))
    i, j, r = edges
    return inv[i], inv[j], r


# ----------------------------------------------------------------------------- model
def _glorot(lin):
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def column_pad(j, N):
    """(N, D) edge indices per column j, padded with E (points at a -inf sentinel)."""
    jj = j.cpu().numpy()
    E = len(jj)
    order = np.argsort(jj, kind="stable")
    cnt = np.bincount(jj, minlength=N)
    D = max(int(cnt.max()), 1)
    pad = np.full((N, D), E, dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    for col in np.nonzero(cnt)[0]:
        pad[col, :cnt[col]] = order[start[col]:start[col] + cnt[col]]
    return torch.from_numpy(pad).to(j.device)


class RSR(nn.Module):
    def __init__(self, U, R):
        super().__init__()
        self.rel = _glorot(nn.Linear(R, 1))
        self.head = _glorot(nn.Linear(U, 1))
        self.tail = _glorot(nn.Linear(U, 1))
        self.fc = _glorot(nn.Linear(2 * U, 1))

    def propagate(self, f, i, j, r, pad=None):
        N = f.shape[0]
        if pad is None:
            pad = column_pad(j, N)
        s = (F.leaky_relu(self.head(f), LEAK)[i, 0] + F.leaky_relu(self.tail(f), LEAK)[j, 0]
             + F.leaky_relu(self.rel(r), LEAK)[:, 0])
        smax = torch.cat([s, s.new_full((1,), -torch.inf)])[pad].max(1).values  # per-column max (MPS-safe)
        a = torch.exp(s - smax[j])
        den = torch.zeros(N, device=f.device).index_add(0, j, a)
        w = a / den[j]
        return torch.zeros_like(f).index_add(0, i, w[:, None] * f[j])

    def forward(self, f, i, j, r, pad=None):
        return F.leaky_relu(self.fc(torch.cat([f, self.propagate(f, i, j, r, pad)], 1)), LEAK)


def dense_reference(model, f, rel):
    """The literal TF formula on a dense (N,N,R) tensor (for tests)."""
    with torch.no_grad():
        N = f.shape[0]
        relw = F.leaky_relu(model.rel(torch.from_numpy(rel.astype(np.float32))), LEAK)[..., 0]
        head = F.leaky_relu(model.head(f), LEAK)
        tail = F.leaky_relu(model.tail(f), LEAK)
        weight = head @ torch.ones(1, N) + torch.ones(N, 1) @ tail.T + relw
        mask = torch.where(torch.from_numpy(rel.sum(2) == 0), -1e9, 0.0)
        wm = torch.softmax(mask + weight, 0)
        out = wm @ f
        return F.leaky_relu(model.fc(torch.cat([f, out], 1)), LEAK)


def losses(pred, base, gt, m, alpha):
    rr = (pred - base) / base
    nz = (m != 0).sum().clamp(min=1)
    reg = (m * (rr - gt) ** 2).sum() / nz
    # == relu((rr_i - rr_j)(gt_j - gt_i) m_i m_j).mean() over all N^2 pairs; masked rows contribute 0
    k = m[:, 0] > 0
    rs, gs = rr[k], gt[k]
    rank = F.relu((rs - rs.T) * (gs.T - gs)).sum() / (m.shape[0] ** 2)
    return reg + alpha * rank, reg, rank, rr


# ----------------------------------------------------------------------------- evaluation (evaluator.py)
def daily_metrics(pred, gt, mask):
    """Per test day: squared-error sum and count, top-1 realised return, reciprocal rank of the
    predicted top-1, mean top-5 / top-10 return. Same selection rules as evaluator.evaluate."""
    N, D = pred.shape
    out = {k: np.zeros(D) for k in ("se", "cnt", "top1", "rr", "top5", "top10")}
    for d in range(D):
        m = mask[:, d] >= 0.5
        out["se"][d] = float((((pred[:, d] - gt[:, d]) * mask[:, d]) ** 2).sum())
        out["cnt"][d] = float(mask[:, d].sum())
        order_p = np.argsort(pred[:, d])[::-1]
        order_p = order_p[m[order_p]]
        order_g = np.argsort(gt[:, d])[::-1]
        order_g = order_g[m[order_g]]
        top1 = order_p[0]
        out["top1"][d] = gt[top1, d]
        out["top5"][d] = gt[order_p[:5], d].sum() / 5
        out["top10"][d] = gt[order_p[:10], d].sum() / 10
        out["rr"][d] = 1.0 / (int(np.where(order_g == top1)[0][0]) + 1)
    return out


def summary(dm):
    return {"mse": float(dm["se"].sum() / dm["cnt"].sum()), "mrrt": float(dm["rr"].mean()),
            "irr_sum": float(dm["top1"].sum()), "btl": float(1.0 + dm["top1"].sum())}


# ----------------------------------------------------------------------------- training
class Data:
    def __init__(self, market, emb_fname, seq, mode, steps=1):
        self.mask, self.gt, self.base = load_eod(market, steps)
        self.emb = load_embedding(emb_fname)
        assert self.emb.shape[:2] == (self.mask.shape[0], self.mask.shape[1]), (self.emb.shape, self.mask.shape)
        self.seq, self.steps = seq, steps
        if mode == "confirm":
            self.vi, self.ti, self.T = VALID_INDEX, TEST_INDEX, self.mask.shape[1]
        elif mode == "dev":                       # shift one year back; nothing at/after TEST_INDEX is used
            self.vi, self.ti, self.T = VALID_INDEX - 252, VALID_INDEX, TEST_INDEX
        else:
            raise ValueError(mode)

    def batch(self, off):
        s, h = self.seq, self.steps
        assert off + s + h - 1 < self.T
        m = self.mask[:, off:off + s + h].min(1)
        return (torch.from_numpy(np.ascontiguousarray(self.emb[:, off, :], dtype=np.float32)),
                torch.from_numpy(m[:, None]), torch.from_numpy(self.base[:, off + s - 1][:, None]),
                torch.from_numpy(self.gt[:, off + s + h - 1][:, None]))

    def offsets(self, part):
        s, h = self.seq, self.steps
        if part == "val":
            return range(self.vi - s - h + 1, self.ti - s - h + 1)
        if part == "test":
            return range(self.ti - s - h + 1, self.T - s - h + 1)
        raise ValueError(part)


def run_model(data: Data, edges, R, U, alpha, seed, epochs=50, lr=1e-3, eval_test=True, log=None, device="cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i, j, r = (torch.from_numpy(np.asarray(x)) for x in edges)
    i, j, r = i.long().to(device), j.long().to(device), r.to(device)
    model = RSR(U, R).to(device)
    pad = column_pad(j, data.mask.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    n_train = data.vi - data.seq - data.steps + 1
    val_off = list(data.offsets("val"))
    dev = lambda bt: tuple(x.to(device) for x in bt)  # noqa: E731
    val_cache = [dev(data.batch(o)) for o in val_off]
    best = (np.inf, None, -1)
    for ep in range(epochs):
        model.train()
        order = np.arange(data.vi)
        rng.shuffle(order)
        for o in order[:n_train]:
            f, m, b, g = dev(data.batch(int(o)))
            loss = losses(model(f, i, j, r, pad), b, g, m, alpha)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(float(losses(model(f, i, j, r, pad), b, g, m, alpha)[0]) for f, m, b, g in val_cache)
        vl /= (data.ti - data.vi)
        if vl < best[0]:
            best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        if log:
            log(f"  ep {ep} val_loss {vl:.6e}")
    model.load_state_dict(best[1])
    model.eval()
    res = {"best_epoch": best[2], "val_loss": best[0]}
    parts = ["val", "test"] if eval_test else ["val"]
    with torch.no_grad():
        for part in parts:
            P, G, M = [], [], []
            for o in data.offsets(part):
                f, m, b, g = dev(data.batch(o))
                rr = losses(model(f, i, j, r, pad), b, g, m, alpha)[3]
                P.append(rr[:, 0].cpu().numpy()); G.append(g[:, 0].cpu().numpy()); M.append(m[:, 0].cpu().numpy())
            res[part] = daily_metrics(np.stack(P, 1), np.stack(G, 1), np.stack(M, 1))
    return res
