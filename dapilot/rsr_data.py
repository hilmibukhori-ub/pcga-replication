"""RSR loaders (PREREG_RSR.md). Graph structure and industry labels only; prices are loaded by the
pipeline. Clarifications C1 (symmetrised Wiki, no diagonal) and C2 (`n/a` -> singleton industries)
from DEVIATIONS.md are implemented here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data/raw/rsr_repo"
REL = ROOT / "data/raw/rsr_relation/relation"
NA_LABEL = "n/a"


def tickers(market: str) -> list[str]:
    return np.genfromtxt(REPO / f"data/{market}_tickers_qualify_dr-0.98_min-5_smooth.csv", dtype=str).tolist()


def wiki_edges(market: str) -> np.ndarray:
    """Undirected Wiki edges (u < v), any relation channel in either direction, no self-loops (C1)."""
    r = np.load(REL / f"wikidata/{market}_wiki_relation.npy")
    A = r.sum(2) > 0
    A = A | A.T
    np.fill_diagonal(A, False)
    return np.argwhere(np.triu(A, 1))


def industry_blocks(market: str) -> np.ndarray:
    """Integer industry id per stock in ticker order; each `n/a` stock gets its own id (C2)."""
    tick = tickers(market)
    pos = {t: i for i, t in enumerate(tick)}
    ind = json.loads((REL / f"sector_industry/{market}_industry_ticker.json").read_text())
    blocks = np.full(len(tick), -1)
    nxt = 0
    for g in sorted(ind):
        members = [pos[t] for t in ind[g] if t in pos]
        if g == NA_LABEL:
            for i in members:
                blocks[i] = nxt
                nxt += 1
        elif members:
            blocks[members] = nxt
            nxt += 1
    if (blocks < 0).any():
        raise ValueError(f"{market}: {(blocks < 0).sum()} stocks without industry label")
    return blocks
