"""Prospective cross-model construct recovery for PCGA-L.

The experiment reuses the controlled ladder world but fits three matched expert
families: ridge, pooled MLP, and a two-layer message-passing GCN.  Every family has
the same input dimension / parameterisation at L0--L4.  At L0 the graph channels or
message operators are exactly zero, so a difference cannot be caused by extra model
capacity.  Neural initialisation and minibatch order are paired across graph levels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .ladder import generate, graph_features, placebo_set
from .pcga_l import decompose_loss_levels


FAMILIES = ("ridge", "mlp", "gcn")
GRAPH_LEVELS = ("L0_none", "L1_er", "L2_degree", "L3_block", "L4_real")


def split_slices(cfg):
    """Decision-time slices with the final label of every split purged."""
    return {name: slice(bounds[0], bounds[1] - 1) for name, bounds in cfg["splits"].items()}


def matched_graph_features(x: np.ndarray, A: np.ndarray | None) -> np.ndarray:
    """Own, neighbour-mean and degree-scaled-sum channels at every level.

    L0 keeps the same six columns as graph levels but sets both graph-channel groups
    to zero.  This is the matched version of the original transparent ridge input.
    """
    if A is None:
        return np.concatenate([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
    return graph_features(x, A).astype(np.float32)


def graph_operators(A: np.ndarray | None, n: int):
    """Dense mean and degree-scaled-sum operators; both are zero at L0."""
    if A is None:
        z = np.zeros((n, n), dtype=np.float32)
        return z, z.copy()
    adj = A.astype(np.float32)
    degree = adj.sum(1)
    mean = adj / np.maximum(degree, 1.0)[:, None]
    degree_sum = adj / max(float(degree.mean()), 1e-12)
    return mean.astype(np.float32), degree_sum.astype(np.float32)


def _standardise(F: np.ndarray, train_slice: slice):
    reference = F[train_slice].reshape(-1, F.shape[-1])
    mu, sd = reference.mean(0), reference.std(0)
    sd[sd < 1e-8] = 1.0
    return ((F - mu) / sd).astype(np.float32)


def fit_eval_ridge(F: np.ndarray, y: np.ndarray, cfg: dict):
    slices = split_slices(cfg)
    F = _standardise(F, slices["train"])

    def flat(part):
        return F[part].reshape(-1, F.shape[-1]), y[part].reshape(-1)

    Xtr, ytr = flat(slices["train"])
    Xva, yva = flat(slices["val"])
    best = None
    for alpha in cfg["models"]["ridge"]["alpha_grid"]:
        beta = np.linalg.solve(
            Xtr.T @ Xtr + float(alpha) * np.eye(Xtr.shape[1]), Xtr.T @ ytr
        )
        val = float(np.mean((Xva @ beta - yva) ** 2))
        if best is None or val < best[0]:
            best = (val, float(alpha), beta)
    Xte, yte = flat(slices["test"])
    loss = float(np.mean((Xte @ best[2] - yte) ** 2))
    return loss, {"selected_alpha": best[1], "val_mse": best[0]}


class PooledMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MatchedGraphNet(nn.Module):
    """Two-layer GCN with self, neighbour-mean and neighbour-sum pathways."""

    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.self1 = nn.Linear(d_in, hidden)
        self.mean1 = nn.Linear(d_in, hidden, bias=False)
        self.sum1 = nn.Linear(d_in, hidden, bias=False)
        self.self2 = nn.Linear(hidden, hidden)
        self.mean2 = nn.Linear(hidden, hidden, bias=False)
        self.sum2 = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, mean_op, sum_op):
        mean_x = torch.matmul(mean_op, x)
        sum_x = torch.matmul(sum_op, x)
        h = torch.relu(self.self1(x) + self.mean1(mean_x) + self.sum1(sum_x))
        mean_h = torch.matmul(mean_op, h)
        sum_h = torch.matmul(sum_op, h)
        h = torch.relu(self.self2(h) + self.mean2(mean_h) + self.sum2(sum_h))
        return self.out(h).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _train_neural(
    family: str,
    inputs: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    seed: int,
    A: np.ndarray | None = None,
):
    slices = split_slices(cfg)
    model_cfg = cfg["models"][family]
    x = _standardise(inputs, slices["train"])
    y_scale = float(y[slices["train"]].std())
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        raise ValueError("training target has zero or non-finite scale")
    ys = (y / y_scale).astype(np.float32)
    xt, yt = torch.from_numpy(x), torch.from_numpy(ys)
    torch.manual_seed(seed)
    if family == "mlp":
        model = PooledMLP(x.shape[-1], int(model_cfg["hidden"]))
        operators = None
    elif family == "gcn":
        model = MatchedGraphNet(x.shape[-1], int(model_cfg["hidden"]))
        mean, degree_sum = graph_operators(A, x.shape[1])
        operators = torch.from_numpy(mean), torch.from_numpy(degree_sum)
    else:
        raise ValueError(f"unknown neural family: {family}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_cfg["lr"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    train_idx = torch.arange(slices["train"].start, slices["train"].stop)
    val_idx = torch.arange(slices["val"].start, slices["val"].stop)
    test_idx = torch.arange(slices["test"].start, slices["test"].stop)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, bad = np.inf, None, -1, 0
    tolerance = float(model_cfg["min_delta"])

    def predict(index):
        model.eval()
        with torch.no_grad():
            if family == "mlp":
                return model(xt[index])
            return model(xt[index], *operators)

    for epoch in range(int(model_cfg["max_epochs"])):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for start in range(0, len(order), int(model_cfg["batch_days"])):
            batch = order[start:start + int(model_cfg["batch_days"])]
            if family == "mlp":
                pred = model(xt[batch])
            else:
                pred = model(xt[batch], *operators)
            loss = ((pred - yt[batch]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = float(((predict(val_idx) - yt[val_idx]) ** 2).mean())
        if val_loss < best_loss - tolerance:
            best_loss = val_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_epoch, bad = epoch, 0
        else:
            bad += 1
            if bad >= int(model_cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("neural model never produced a finite validation checkpoint")
    model.load_state_dict(best_state)
    pred = predict(test_idx).numpy() * y_scale
    test_loss = float(np.mean((pred - y[slices["test"]]) ** 2))
    return test_loss, {
        "val_mse_scaled": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "parameters": parameter_count(model),
    }


def _level_graphs(A, sectors, rng, k):
    placebo = placebo_set(A, sectors, rng, k)
    return {
        "L0_none": [None],
        "L1_er": placebo["L1_er"],
        "L2_degree": placebo["L2_degree"],
        "L3_block": placebo["L3_block"],
        "L4_real": [A],
    }


def run_seed(cfg: dict, seed: int):
    """Fit all pre-specified family/scenario/level combinations for one DGP seed."""
    world = generate(cfg, seed)
    rng = np.random.default_rng([int(cfg["experiment"]["graph_seed"]), seed])
    levels = _level_graphs(
        world["A"], world["sectors"], rng,
        int(cfg["experiment"]["placebos_per_level"]),
    )
    feature_cache = {
        level: [matched_graph_features(world["x"], graph) for graph in graphs]
        for level, graphs in levels.items()
    }
    rows, diagnostics = [], []
    scenario_names = tuple(cfg["experiment"]["scenarios"])
    family_names = tuple(cfg["experiment"]["families"])
    for scenario_index, scenario in enumerate(scenario_names):
        y = world["worlds"][scenario]["y"]
        for family_index, family in enumerate(family_names):
            row = {"seed": seed, "scenario": scenario, "family": family}
            paired_seed = int(cfg["experiment"]["model_seed"]) + 10_000 * seed + 100 * scenario_index + family_index
            for level, graphs in levels.items():
                losses = []
                for variant, graph in enumerate(graphs):
                    if family == "ridge":
                        loss, info = fit_eval_ridge(feature_cache[level][variant], y, cfg)
                    elif family == "mlp":
                        loss, info = _train_neural(
                            family, feature_cache[level][variant], y, cfg, paired_seed,
                        )
                    elif family == "gcn":
                        loss, info = _train_neural(
                            family, world["x"].astype(np.float32), y, cfg, paired_seed, A=graph,
                        )
                    else:
                        raise ValueError(f"unknown model family: {family}")
                    losses.append(loss)
                    diagnostics.append({
                        "seed": seed, "scenario": scenario, "family": family,
                        "level": level, "variant": variant, "test_mse": loss, **info,
                    })
                row[level] = float(np.mean(losses))
            rows.append(row)
    return decompose_loss_levels(pd.DataFrame(rows)), pd.DataFrame(diagnostics)


def write_seed_result(rows: pd.DataFrame, diagnostics: pd.DataFrame, out: Path, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / f"seed_{seed:05d}.csv", index=False)
    diagnostics.to_csv(out / f"seed_{seed:05d}_diagnostics.csv", index=False)

