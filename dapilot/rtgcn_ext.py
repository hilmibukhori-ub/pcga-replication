"""Extension of the RT-GCN re-test (PREREG_RTGCN_EXT.md): NYSE, and the Time-aware variant RT-GCN(T).

Builds on the frozen harness `dapilot/rtgcn_replica.py` (imported, never modified): same loss, same data
loading, same training/evaluation loop (`rt.run`), same checkpointing. What is added:

- Market constants for NYSE (verified on the released tensors, see PREREG_RTGCN_EXT.md §2).
- `build_model_ext(edge, relation, variant)`: the ORIGINAL `net.rt_gcn.Model`, with `Graph.get_edge`
  replaced by a version that installs a given edge matrix and a given relation tensor (V x V x D, the
  non-self channels). For variant U the relation is a (1, 1, D) placeholder (U never reads it; as in the
  frozen harness). For variant T the full tensor is installed, because T reads it in two places:
  the attention bias `mask = 1[sum_d relation > 0]` and the relation-importance `Linear(D, 1)`.
  Tests check parameters and outputs against the model built by the original Graph from the file.
- Graphs for T: real (released relations), none (all-zero relation tensor: no relation information, the
  same attention network), perm k (within-industry node permutation of the whole tensor).
- `run_ext`: `rt.run` with the model factory swapped to `build_model_ext` (the loop itself is reused).
"""

from __future__ import annotations

import numpy as np

from . import rtgcn_replica as rt
from net.utils import graph as _graph_mod        # noqa: E402  (path set up by rtgcn_replica)
from net.rt_gcn import Model                       # noqa: E402

N_REL = {"NASDAQ": 136, "NYSE": 134}                       # non-self channels
WIKI = {"NASDAQ": range(0, 40), "NYSE": range(0, 27)}      # rest (up to N_REL) = industry cliques
N_NODES = {"NASDAQ": 854, "NYSE": 1405}
_CUR = {}


def relation(market, channels=None):
    """Non-self relation channels as float32 (V, V, D). channels=None -> all non-self channels."""
    r = np.load(rt.DATA / market / f"{market}_relation.npy", mmap_mode="r")
    idx = list(range(N_REL[market])) if channels is None else list(channels)
    return np.ascontiguousarray(r[:, :, idx], dtype=np.float32)


def edge_from_relation(rel):
    """eye + 1[any relation], exactly as Graph.get_edge."""
    return np.eye(rel.shape[0]) + (rel.sum(2) >= 1).astype(np.float64)


def industry_labels(market):
    return np.load(rt.DATA / market / "industry_labels_derived.npy", allow_pickle=False)


def _get_edge(self, market, relation_path):
    self.num_node = _CUR["edge"].shape[0]
    self.relation = _CUR["relation"]
    self.edge = _CUR["edge"]


def build_model_ext(edge, rel, variant):
    """variant 'U' -> edge_importance_weighting='Uniform', 'T' -> 'Time-aware'.
    rel: (V, V, D) float32 for T; for U only its channel count matters (placeholder installed)."""
    mode = {"U": "Uniform", "T": "Time-aware"}[variant]
    D = rel.shape[2]
    _CUR.update(edge=edge, relation=rel if variant == "T" else np.zeros((1, 1, D)))
    orig = _graph_mod.Graph.get_edge
    _graph_mod.Graph.get_edge = _get_edge
    try:
        m = Model(in_channels=4, graph_args={"market": "NASDAQ", "relation_path": None, "strategy": "uniform"},
                  edge_importance_weighting=mode, dropout=0.5)
    finally:
        _graph_mod.Graph.get_edge = orig
        _CUR.clear()
    return m


def graph_for(market, graph, k=None, seed0=20261111):
    """Returns (edge, relation) for graph in {real, none, ind, perm}."""
    R = relation(market)
    if graph == "real":
        return edge_from_relation(R), R
    if graph == "none":
        V = R.shape[0]
        return np.eye(V), np.zeros_like(R)
    if graph == "ind":
        Ri = R.copy()
        Ri[:, :, list(WIKI[market])] = 0
        return edge_from_relation(Ri), Ri
    if graph == "perm":
        pi = rt.within_label_perm(industry_labels(market), np.random.default_rng([seed0, 7, k]))
        Rp = np.ascontiguousarray(R[np.ix_(pi, pi)])
        return edge_from_relation(Rp), Rp
    raise ValueError(graph)


def run_ext(edge, rel, variant, seed, train, val, test, **kw):
    """rt.run (frozen loop, checkpointing included) with the model factory swapped."""
    orig = rt.build_model
    rt.build_model = lambda e, n_ch: build_model_ext(e, rel, variant)
    try:
        return rt.run(edge, rel.shape[2], seed, train, val, test, **kw)
    finally:
        rt.build_model = orig
