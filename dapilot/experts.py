"""Forecasting experts. Trained once on the expert-train split, early-stopped / tuned on
expert-val, then frozen. Experts see ObservableData only."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from .dgp import ObservableData
from .graphs import mean_aggregation, permute_graph
from .splits import TemporalSplits


# ----------------------------------------------------------------------------- inputs
def build_inputs(obs: ObservableData, kind: str) -> np.ndarray:
    """Per-asset input tensor (T, N, d) using information available at t only."""
    T, N = obs.T, obs.N
    S = int(obs.sectors.max()) + 1
    parts = [
        obs.X,
        np.broadcast_to(obs.factors[:, None, :], (T, N, obs.factors.shape[1])),
        np.broadcast_to(np.eye(S, dtype=np.float32)[obs.sectors][None], (T, N, S)),
    ]
    if kind == "cross":
        mkt = obs.X.mean(1, keepdims=True)
        sec = np.zeros_like(obs.X)
        for s in range(S):
            m = obs.sectors == s
            sec[:, m] = obs.X[:, m].mean(1, keepdims=True)
        parts += [np.broadcast_to(mkt, obs.X.shape), sec]
    elif kind != "base":
        raise ValueError(kind)
    return np.concatenate(parts, -1).astype(np.float32)


def standardise(Z: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    mu = Z[fit_idx].reshape(-1, Z.shape[-1]).mean(0)
    sd = Z[fit_idx].reshape(-1, Z.shape[-1]).std(0)
    sd[sd < 1e-8] = 1.0
    return ((Z - mu) / sd).astype(np.float32)


def label_array(obs: ObservableData, horizon: int) -> np.ndarray:
    """y[t] = returns[t + h]; NaN where the label does not exist."""
    y = np.full_like(obs.returns, np.nan)
    y[:-horizon] = obs.returns[horizon:]
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- models
class GraphNet(nn.Module):
    """Two-layer message-passing net with separate self and neighbour weights
    (mean-aggregation GCN on an explicit adjacency). With use_graph=False it is the matched
    non-graph MLP (identical self pathway, no neighbour pathway)."""

    def __init__(self, d_in: int, hidden: int, use_graph: bool):
        super().__init__()
        self.use_graph = use_graph
        self.self1, self.self2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        if use_graph:
            self.nb1 = nn.Linear(d_in, hidden, bias=False)
            self.nb2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, Z: torch.Tensor, A: torch.Tensor | None) -> torch.Tensor:
        h = self.self1(Z)
        if self.use_graph:
            h = h + self.nb1(torch.bmm(A, Z))
        h = torch.relu(h)
        h2 = self.self2(h)
        if self.use_graph:
            h2 = h2 + self.nb2(torch.bmm(A, h))
        return self.out(torch.relu(h2)).squeeze(-1)


def _predict(model, Zt, At, idx, chunk=256):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            b = idx[i:i + chunk]
            out.append(model(Zt[b], At[b] if At is not None else None))
    return torch.cat(out).numpy()


def train_graphnet(Z, Ahat, y, train_idx, val_idx, ecfg, wd, seed):
    torch.manual_seed(seed)
    model = GraphNet(Z.shape[-1], ecfg["hidden"], Ahat is not None)
    opt = torch.optim.Adam(model.parameters(), lr=ecfg["lr"], weight_decay=wd)
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y)
    At = torch.from_numpy(Ahat) if Ahat is not None else None
    tr = torch.from_numpy(train_idx)
    gen = torch.Generator().manual_seed(seed)
    bs = ecfg["batch_size"]
    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(ecfg["max_epochs"]):
        model.train()
        order = tr[torch.randperm(len(tr), generator=gen)]
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            loss = ((model(Zt[b], At[b] if At is not None else None) - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = float(((_predict(model, Zt, At, val_idx) - y[val_idx]) ** 2).mean())
        if val < best - 1e-6:
            best, best_state, bad, best_ep = val, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= ecfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, best_ep


# ----------------------------------------------------------------------------- orchestration
def train_all_experts(obs: ObservableData, splits: TemporalSplits, cfg: dict, seed: int, log):
    """Returns (preds: name -> (T, N) float32 with NaN at undefined t, info, placebo perms)."""
    ecfg = cfg["experts"]
    h = splits.horizon
    y = label_array(obs, h)
    tr, va = splits.decision_times("expert_train"), splits.decision_times("expert_val")
    splits.assert_labels_within(tr, "expert_train")
    splits.assert_labels_within(va, "expert_val")
    assert np.isfinite(y[tr]).all() and np.isfinite(y[va]).all()
    all_t = np.arange(obs.T - h)

    Zb = standardise(build_inputs(obs, "base"), tr)
    Zc = standardise(build_inputs(obs, "cross"), tr)
    preds, info = {}, {}

    # Ridge (pooled linear, non-graph)
    Xtr, Xva = Zb[tr].reshape(-1, Zb.shape[-1]), Zb[va].reshape(-1, Zb.shape[-1])
    best = None
    for a in ecfg["ridge_alpha_grid"]:
        m = Ridge(alpha=a).fit(Xtr, y[tr].ravel())
        v = float(((m.predict(Xva) - y[va].ravel()) ** 2).mean())
        if best is None or v < best[0]:
            best = (v, a, m)
    p = np.full(obs.returns.shape, np.nan, dtype=np.float32)
    p[all_t] = best[2].predict(Zb[all_t].reshape(-1, Zb.shape[-1])).reshape(len(all_t), -1)
    preds["ridge"], info["ridge"] = p, {"alpha": best[1], "val_mse": best[0]}
    log(f"expert ridge: alpha={best[1]} val_mse={best[0]:.4f}")

    rng = np.random.default_rng([seed, 777])
    perms = [rng.permutation(obs.N) for _ in range(ecfg["n_placebo"])]
    specs = [("mlp", Zb, None), ("cross_mlp", Zc, None), ("gcn", Zb, obs.A_obs)]
    specs += [(f"placebo_{k}", Zb, permute_graph(obs.A_obs, pk)) for k, pk in enumerate(perms)]

    for j, (name, Z, A) in enumerate(specs):
        Ahat = mean_aggregation(A) if A is not None else None
        best = None
        for w_i, wd in enumerate(ecfg["weight_decay_grid"]):
            mseed = seed * 100_000 + j * 100 + w_i
            model, v, ep = train_graphnet(Z, Ahat, y, tr, va, ecfg, wd, mseed)
            if best is None or v < best[0]:
                best = (v, wd, model, ep)
        p = np.full(obs.returns.shape, np.nan, dtype=np.float32)
        At = torch.from_numpy(Ahat) if Ahat is not None else None
        p[all_t] = _predict(best[2], torch.from_numpy(Z), At, all_t)
        preds[name] = p
        info[name] = {"weight_decay": best[1], "val_mse": best[0], "best_epoch": best[3]}
        log(f"expert {name}: wd={best[1]} val_mse={best[0]:.4f} epoch={best[3]}")
    return preds, info, perms
