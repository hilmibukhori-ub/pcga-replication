"""RSR secondary expert robustness (PREREG_RSR.md §5, §9.4): matched 2-layer GCN vs matched MLP.

EXPLORATORY / DESCRIPTIVE. Not part of the confirmatory family. The test period has already been
read once by the confirmatory run (outputs/rsr_ladder_v1); this module never touches that output.

Data preparation, masks, winsorisation, features and placebo graphs are identical to
`rsr_pipeline.run_market`: the same helper functions, and the same RNG with the same call order,
so the L1/L2 placebos and the first K L3 placebos are the same graphs as in the confirmatory run.

Model: dapilot.experts.GraphNet architecture (separate self / neighbour weights, mean aggregation,
no self loops), re-implemented with a sparse static adjacency and a masked loss. MLP = the same
network without the neighbour pathway. The target is divided by its train-period sd for
optimisation (otherwise the 1e-6 early-stopping threshold exceeds the MSE itself); predictions are
rescaled before any loss is computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .power import adjacency, degree_edges, er_edges, industry_edges
from .rsr_data import industry_blocks, wiki_edges
from .rsr_pipeline import (TEST_START, dev_split, features, load_close, split_rows, target,
                           verify_freeze)

MARKETS = ["NASDAQ", "NYSE"]


# ----------------------------------------------------------------------------- data (mirrors run_market)
def prepare(cfg, market, mode):
    split = cfg["data"]["split"]
    if mode == "dev":
        sp_, max_date = dev_split(split), split["val"][1]
    elif mode == "confirm":
        verify_freeze()
        sp_, max_date = split, split["test"][1]
    else:
        raise ValueError(mode)
    C, dates = load_close(market, max_date)
    if mode == "dev":
        assert dates[-1] < TEST_START, "dev mode must not load test rows"
    rows = split_rows(dates, sp_)
    lb = cfg["task"]["min_lookback_days"]
    X = features(C, lb)
    Yraw, M = target(C)
    M[:lb] = False
    lo, hi = rows["train"]
    ql, qh = np.quantile(Yraw[lo:hi + 1][M[lo:hi + 1]], cfg["task"]["winsorise_quantiles_from_train"])
    Y = np.clip(Yraw, ql, qh)
    X = np.nan_to_num(X)
    return dict(X=X, Y=Y, Yraw=Yraw, M=M, rows=rows, dates=dates, n=C.shape[1])


def graphs(cfg, market, n, k_l3):
    """Same RNG and call order as rsr_pipeline.run_market -> identical placebo graphs."""
    E, blocks = wiki_edges(market), industry_blocks(market)
    rng = np.random.default_rng([cfg["experiment"]["seed"], MARKETS.index(market)])
    kp = cfg["graphs"]["placebos"]
    g = {"L0_none": None, "L4_real": adjacency(E, n)}
    for k in range(kp["L1_er"]):
        g[f"L1_er_{k}"] = adjacency(er_edges(rng, n, len(E)), n)
    for k in range(kp["L2_degree"]):
        g[f"L2_degree_{k}"] = adjacency(degree_edges(E, rng), n)
    for k in range(k_l3):
        g[f"L3_industry_{k}"] = adjacency(industry_edges(E, blocks, rng), n)
    return g


def sparse_mean_agg(A):
    """Row-normalised D^-1 A as a torch sparse COO tensor; isolated nodes get a zero row."""
    A = A.tocsr().astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    A = A.multiply(1.0 / np.maximum(deg, 1.0)[:, None]).tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(A.data), A.shape).coalesce()


# ----------------------------------------------------------------------------- model
def agg(A, H):
    """Neighbour mean for a batch of days: H (B, N, d) -> (B, N, d)."""
    B, N, d = H.shape
    out = torch.sparse.mm(A, H.permute(1, 0, 2).reshape(N, B * d))
    return out.reshape(N, B, d).permute(1, 0, 2)


class StaticGraphNet(nn.Module):
    def __init__(self, d_in, hidden, use_graph):
        super().__init__()
        self.use_graph = use_graph
        self.self1, self.self2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        if use_graph:
            self.nb1 = nn.Linear(d_in, hidden, bias=False)
            self.nb2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, Z, A):
        h = self.self1(Z)
        if self.use_graph:
            h = h + self.nb1(agg(A, Z))
        h = torch.relu(h)
        h2 = self.self2(h)
        if self.use_graph:
            h2 = h2 + self.nb2(agg(A, h))
        return self.out(torch.relu(h2)).squeeze(-1)


def _masked_mse(p, y, m):
    return ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def _predict(model, Z, A, t_idx, chunk=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(t_idx), chunk):
            out.append(model(Z[t_idx[i:i + chunk]], A))
    return torch.cat(out).numpy()


def train_one(Z, A, ys, Mt, tr, va, ecfg, wd, seed):
    torch.manual_seed(seed)
    model = StaticGraphNet(Z.shape[-1], ecfg["hidden"], A is not None)
    opt = torch.optim.Adam(model.parameters(), lr=ecfg["lr"], weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    trt = torch.from_numpy(tr)
    best, state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(ecfg["max_epochs"]):
        model.train()
        order = trt[torch.randperm(len(trt), generator=gen)]
        for i in range(0, len(order), ecfg["batch_size"]):
            b = order[i:i + ecfg["batch_size"]]
            loss = _masked_mse(model(Z[b], A), ys[b], Mt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = torch.from_numpy(_predict(model, Z, A, va))
        v = float(_masked_mse(pv, ys[va], Mt[va]))
        if v < best - 1e-6:
            best, state, bad, best_ep = v, {k: x.clone() for k, x in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= ecfg["patience"]:
                break
    model.load_state_dict(state)
    return model, best, best_ep


def daily_losses(pred, Y, Yraw, M):
    cnt = M.sum(1)
    d = lambda e: np.where(M, e, 0.0).sum(1) / cnt  # noqa: E731
    ic = []
    for t in range(len(pred)):
        m = M[t]
        p, q = pred[t][m], Yraw[t][m]
        ic.append(np.corrcoef(p, q)[0, 1] if m.sum() > 2 and p.std() > 0 else np.nan)
    return {"mse": d((pred - Y) ** 2), "mse_raw": d((pred - Yraw) ** 2), "mae": d(np.abs(pred - Yraw)),
            "ic": np.array(ic)}


# ----------------------------------------------------------------------------- one (market, seed)
def run(cfg, ecfg, market, mode, seed_idx, k_l3, out: Path, log=print, only=None):
    D = prepare(cfg, market, mode)
    X, Y, Yraw, M, rows = D["X"], D["Y"], D["Yraw"], D["M"], D["rows"]
    tr = np.arange(rows["train"][0], rows["train"][1] + 1)
    va = np.arange(rows["val"][0], rows["val"][1] + 1)
    te = np.arange(rows["test"][0], rows["test"][1] + 1)
    trm = M[tr]
    mu, sd = X[tr][trm].mean(0), X[tr][trm].std(0) + 1e-12
    Z = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    s = float(Y[tr][trm].std())
    ys = torch.from_numpy(np.where(M, Y / s, 0.0).astype(np.float32))
    Mt = torch.from_numpy(M.astype(np.float32))
    G = graphs(cfg, market, D["n"], k_l3)
    out.mkdir(parents=True, exist_ok=True)
    for j, (name, Agr) in enumerate(G.items()):
        if only and not any(name.startswith(o) for o in only):
            continue
        f = out / f"{market}_s{seed_idx}_{name}.npz"
        if f.exists():
            continue
        A = sparse_mean_agg(Agr) if Agr is not None else None
        best = None
        for w_i, wd in enumerate(ecfg["weight_decay_grid"]):
            mseed = cfg["experiment"]["seed"] + seed_idx * 100_000 + j * 100 + w_i
            model, v, ep = train_one(Z, A, ys, Mt, tr, va, ecfg, wd, mseed)
            if best is None or v < best[0]:
                best = (v, wd, model, ep)
        pred = _predict(best[2], Z, A, te) * s
        L = daily_losses(pred, Y[te], Yraw[te], M[te])
        np.savez(f, **L, val_mse_scaled=best[0], wd=best[1], best_epoch=best[3],
                 test_first=D["dates"][te[0]], test_last=D["dates"][te[-1]])
        log(f"{market} s{seed_idx} {name}: wd={best[1]} val={best[0]:.5f} ep={best[3]} "
            f"test_mse={L['mse'].mean():.6e} ic={np.nanmean(L['ic']):+.4f}")
