"""Synthetic data-generating process.

Hidden quantities (true graph, regime, corruption level, conditional mean) live in
``HiddenData`` and must never be passed to experts or selectors. Predictors receive only
``ObservableData``.

Model (t = decision time, prediction target is returns[t+1]):
    X_t   = rho_own X_{t-1} + rho_graph W_true X_{t-1} + sigma_x eta_t       (node features)
    f_t   = rho_f f_{t-1} + sqrt(1-rho_f^2) xi_t                             (market factors)
    mu_t  = own(X_t) + C_{sector} f_t + g(regime_t) * (W_true X_t) w_rel      (conditional mean)
    r_{t+1} = mu_t + sigma_c u_{t+1} + sigma_e e_{t+1}                        (returns)
Observed graph: A_true (regimes informative / no_signal) or a rewired A_true (corrupted),
plus per-step edge-observation noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graphs import mean_aggregation, rewire, symmetric_from_upper

REGIME_NAMES = ("informative", "no_signal", "corrupted")
INFORMATIVE, NO_SIGNAL, CORRUPTED = 0, 1, 2


@dataclass(frozen=True)
class ObservableData:
    X: np.ndarray          # (T, N, F) features observed at end of t
    factors: np.ndarray    # (T, K)
    A_obs: np.ndarray      # (T, N, N) bool observed adjacency at t
    sectors: np.ndarray    # (N,) public sector classification
    returns: np.ndarray    # (T, N) returns[t] realised at t; returns[0] = nan

    @property
    def T(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]


@dataclass(frozen=True)
class HiddenData:
    A_true: np.ndarray       # (N, N)
    mu: np.ndarray           # (T, N) mu[t] = E[returns[t+1] | info at t]; mu[T-1] = nan
    regime: np.ndarray       # (T,) int
    corruption: np.ndarray   # (T,) fraction of rewired edges (0 outside corrupted regime)
    segment: np.ndarray      # (T,) segment id


@dataclass(frozen=True)
class SyntheticWorld:
    observable: ObservableData
    hidden: HiddenData
    seed: int


def _unit(rng, d):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def _standardise_from_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Scale ``values`` with constants fitted on a pre-sample reference period.

    The reference period must be fixed before the reported sample starts.  In particular,
    fitting these constants on the complete simulated path would make ``mu[t]`` depend on
    future feature realisations and invalidate its conditional-mean interpretation.
    """
    mean = reference.mean()
    std = reference.std()
    if not np.isfinite(std) or std < 1e-12:
        raise ValueError("pre-sample signal has zero or non-finite variance")
    return (values - mean) / std


def sample_true_graph(rng, sectors, p_intra, p_inter) -> np.ndarray:
    n = len(sectors)
    same = sectors[:, None] == sectors[None, :]
    P = np.where(same, p_intra, p_inter)
    iu = np.triu_indices(n, 1)
    A = symmetric_from_upper(rng.random(len(iu[0])) < P[iu], n)
    for i in range(n):  # no isolated nodes
        if not A[i].any():
            cand = np.flatnonzero(same[i] & (np.arange(n) != i))
            j = rng.choice(cand)
            A[i, j] = A[j, i] = True
    return A


def sample_regimes(rng, T, lmin, lmax, n_regimes=3):
    regime = np.empty(T, dtype=int)
    segment = np.empty(T, dtype=int)
    t, prev, s = 0, -1, 0
    while t < T:
        L = int(rng.integers(lmin, lmax + 1))
        r = int(rng.choice([k for k in range(n_regimes) if k != prev]))
        regime[t:t + L] = r
        segment[t:t + L] = s
        t, prev, s = t + L, r, s + 1
    return regime, segment


def generate_world(cfg: dict, seed: int) -> SyntheticWorld:
    d = cfg["data"]
    # Independent deterministic streams keep a generated prefix invariant when n_steps changes.
    # That property is both a useful causality check and protection against accidental coupling
    # through a variable number of random draws in another component.
    rng_graph = np.random.default_rng([seed, 20261108, 1])
    rng_x = np.random.default_rng([seed, 20261108, 2])
    rng_f = np.random.default_rng([seed, 20261108, 3])
    rng_regime = np.random.default_rng([seed, 20261108, 4])
    rng_obs = np.random.default_rng([seed, 20261108, 5])
    rng_signal = np.random.default_rng([seed, 20261108, 6])
    rng_return = np.random.default_rng([seed, 20261108, 7])
    N, S, T, F, K = d["n_assets"], d["n_sectors"], d["n_steps"], d["n_features"], d["n_factors"]
    burn = d["burn_in"]
    if burn < 4:
        raise ValueError("burn_in must be at least 4 to fit pre-sample signal scales")
    sectors = np.repeat(np.arange(S), N // S)

    A_true = sample_true_graph(
        rng_graph, sectors, d["graph"]["p_intra"], d["graph"]["p_inter"]
    )
    W = mean_aggregation(A_true).astype(np.float64)

    # --- node features diffusing on the true graph
    ro, rg = d["features"]["rho_own"], d["features"]["rho_graph"]
    sig_x = np.sqrt(1 - (ro + rg) ** 2)
    X_full = np.zeros((T + burn, N, F))
    for t in range(1, T + burn):
        X_full[t] = (ro * X_full[t - 1] + rg * W @ X_full[t - 1]
                     + sig_x * rng_x.normal(size=(N, F)))
    pre = slice(burn // 2, burn)  # stationary pre-sample reference, never part of a split
    X_pre, X = X_full[pre], X_full[burn:]

    # --- market factors
    rf = d["factors"]["rho"]
    f_full = np.zeros((T + burn, K))
    for t in range(1, T + burn):
        f_full[t] = rf * f_full[t - 1] + np.sqrt(1 - rf**2) * rng_f.normal(size=K)
    f_pre, f = f_full[pre], f_full[burn:]

    # --- regimes and observed graphs
    regime, segment = sample_regimes(
        rng_regime, T, d["regimes"]["segment_len_min"], d["regimes"]["segment_len_max"]
    )
    corruption = np.zeros(T)
    iu = np.triu_indices(N, 1)
    A_obs = np.zeros((T, N, N), dtype=bool)
    p_drop = d["observation"]["edge_drop"]
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        if regime[idx[0]] == CORRUPTED:
            c = rng_obs.uniform(d["corruption"]["min"], d["corruption"]["max"])
            base = rewire(A_true, c, rng_obs)
            corruption[idx] = c
        else:
            base = A_true
        bvec = base[iu]
        n_e, n_non = bvec.sum(), (~bvec).sum()
        p_add = d["observation"]["edge_add_frac"] * n_e / n_non
        U = rng_obs.random((len(idx), len(bvec)))
        obs = np.where(bvec[None], U >= p_drop, U < p_add)
        for k, t in enumerate(idx):
            A_obs[t] = symmetric_from_upper(obs[k], N)

    # --- conditional mean
    sg = d["signal"]
    w_nl = _unit(rng_signal, F)
    w_lin = _unit(rng_signal, F)
    w_rel = _unit(rng_signal, F)
    nl_pre, nl_raw = np.tanh(1.5 * X_pre @ w_nl), np.tanh(1.5 * X @ w_nl)
    lin_pre, lin_raw = X_pre @ w_lin, X @ w_lin
    nl = _standardise_from_reference(nl_pre, nl_raw)
    lin = _standardise_from_reference(lin_pre, lin_raw)
    share = sg["own_nonlinear_share"]
    nl_pre_z = _standardise_from_reference(nl_pre, nl_pre)
    lin_pre_z = _standardise_from_reference(lin_pre, lin_pre)
    own_pre = np.sqrt(share) * nl_pre_z + np.sqrt(1 - share) * lin_pre_z
    own_raw = np.sqrt(share) * nl + np.sqrt(1 - share) * lin
    own = sg["own_std"] * _standardise_from_reference(own_pre, own_raw)
    C = rng_signal.normal(size=(S, K))
    fac_pre = f_pre @ C[sectors].T
    fac = sg["factor_std"] * _standardise_from_reference(fac_pre, f @ C[sectors].T)
    rel_pre = np.einsum("ij,tjf->tif", W, X_pre) @ w_rel
    rel_raw = _standardise_from_reference(
        rel_pre, np.einsum("ij,tjf->tif", W, X) @ w_rel
    )
    gains = np.array([d["relational_gain"][n] for n in REGIME_NAMES])
    rel = sg["relational_std"] * gains[regime][:, None] * rel_raw
    mu = own + fac + rel

    # --- returns: r[t+1] = mu[t] + noise
    returns = np.full((T, N), np.nan)
    noise_draws = rng_return.normal(size=(T - 1, N + 1))
    noise = (sg["common_noise_std"] * noise_draws[:, :1]
             + sg["idio_noise_std"] * noise_draws[:, 1:])
    returns[1:] = mu[:-1] + noise
    mu = mu.copy()
    mu[-1] = np.nan

    obs_data = ObservableData(
        X=X.astype(np.float32), factors=f.astype(np.float32), A_obs=A_obs,
        sectors=sectors, returns=returns,
    )
    hidden = HiddenData(A_true=A_true, mu=mu, regime=regime, corruption=corruption, segment=segment)
    return SyntheticWorld(observable=obs_data, hidden=hidden, seed=seed)
