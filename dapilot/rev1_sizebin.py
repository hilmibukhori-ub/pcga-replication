"""Revision 1, study S (PREREG_REV1.md): size measure and industry x size-bin placebos for the RSR re-test.

Imports the frozen dapilot/rsr_replica.py and does not modify it. Everything here is return-free: the size
measure uses raw Close x Volume over the RSR training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import rsr_replica as rr

GF = rr.REPO / "data/google_finance"
TAG_PARENT, TAG_SIZEBIN = 7, 8


def training_dates(market, rows=(29, 785)):
    """Trading dates of processed rows 0..755 (rows 29..784 of aver_line_dates; pad_begin = 29 in eod.py)."""
    d = np.genfromtxt(rr.REPO / f"data/{market}_aver_line_dates.csv", dtype=str, delimiter=",")
    return [str(x)[:19] for x in d[rows[0]:rows[1]]]


def size_from_frame(df, window, min_days=50):
    """Median Close*Volume over dates in `window` with Volume > 0; -inf if fewer than min_days such days.
    df: index = raw date strings (timezone suffix allowed), columns include Close and Volume."""
    idx = pd.Index([str(x)[:19] for x in df.index])
    keep = idx.isin(set(window)) & (df["Volume"].to_numpy() > 0)
    dv = (df["Close"].to_numpy() * df["Volume"].to_numpy())[keep]
    return float(np.median(dv)) if len(dv) >= min_days else -np.inf


def size_measure(market, min_days=50):
    window = training_dates(market)
    return np.array([size_from_frame(pd.read_csv(GF / f"{market}_{t}_30Y.csv", index_col=0), window, min_days)
                     for t in rr.tickers(market)])


def sizebin_labels(parent_labels, size, bins=2):
    """Split each parent permutation group into 2 bins: lower floor(n/2) by (size, index) -> 0, rest -> 1.
    Returns combined labels parent*2 + bin."""
    assert bins == 2
    parent_labels = np.asarray(parent_labels)
    b = np.zeros(len(parent_labels), int)
    for g in np.unique(parent_labels):
        idx = np.where(parent_labels == g)[0]
        order = idx[np.lexsort((idx, size[idx]))]
        b[order[len(idx) // 2:]] = 1
    return parent_labels * 2 + b


def sizebin_pi(labels_sb, seed0, mi, k):
    return rr.within_label_perm(labels_sb, np.random.default_rng([seed0, mi, TAG_SIZEBIN, k]))


def parent_pi(parent_labels, seed0, mi, k):
    return rr.within_label_perm(parent_labels, np.random.default_rng([seed0, mi, TAG_PARENT, k]))


def wiki_degree(edges, N):
    i, j, _ = edges
    m = i != j
    A = np.zeros((N, N), bool)
    A[i[m], j[m]] = True
    A |= A.T
    return A.sum(1)


def descriptives(market, mi, seed0, K, size=None):
    """S.7 (return-free): Spearman(degree, size), edge overlap, movable share; real vs parent vs size-bin."""
    from scipy.stats import spearmanr
    size = size_measure(market) if size is None else size
    rel = rr.load_relation(market, "wikidata")
    E = rr.to_edges(rel)
    N = rel.shape[0]
    del rel
    lab = rr.industry_labels(market)
    lab_sb = sizebin_labels(lab, size)
    deg = wiki_degree(E, N)
    fin = np.isfinite(size)
    real_keep = E[0] != E[1]
    real_set = set(zip(E[0][real_keep].tolist(), E[1][real_keep].tolist()))

    def summ(pis):
        rho, ov = [], []
        for pi in pis:
            P = rr.permute_edges(E, pi)
            rho.append(spearmanr(wiki_degree(P, N)[fin], size[fin])[0])
            k_ = P[0] != P[1]
            ov.append(np.mean([e in real_set for e in zip(P[0][k_].tolist(), P[1][k_].tolist())]))
        return {"spearman_deg_size_mean": float(np.mean(rho)), "edge_overlap_mean": float(np.mean(ov))}

    linked = deg > 0
    cnt = {g: (lab_sb == g).sum() for g in np.unique(lab_sb)}
    return {"real_spearman_deg_size": float(spearmanr(deg[fin], size[fin])[0]),
            "parent": summ([parent_pi(lab, seed0, mi, k) for k in range(K)]),
            "sizebin": summ([sizebin_pi(lab_sb, seed0, mi, k) for k in range(K)]),
            "linked_movable_share_sizebin": float(np.mean([cnt[g] > 1 for g in lab_sb[linked]])),
            "n_cells_sizebin": int(len(cnt)), "n_inf_size": int((~fin).sum())}
