"""Configuration loading, validation and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REQUIRED_SECTIONS = ("experiment", "data", "splits", "experts", "selectors", "evaluation")


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate_config(cfg: dict) -> None:
    for s in REQUIRED_SECTIONS:
        if s not in cfg:
            raise ValueError(f"config missing section '{s}'")
    d = cfg["data"]
    if d["n_assets"] % d["n_sectors"] != 0:
        raise ValueError("n_assets must be divisible by n_sectors")
    if len(d["regimes"]["names"]) != 3:
        raise ValueError("exactly three regimes are expected")
    if d["features"]["rho_own"] + d["features"]["rho_graph"] >= 1:
        raise ValueError("feature process must be stationary (rho_own + rho_graph < 1)")
    if not 0 <= d["corruption"]["min"] <= d["corruption"]["max"] <= 1:
        raise ValueError("invalid corruption range")
    if cfg["splits"]["horizon"] < 1:
        raise ValueError("horizon must be >= 1")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash all result-defining fields, including the exact seed set.

    Only execution/location settings are excluded.  Including seeds prevents a subset run from
    silently reusing and aggregating a directory created for a different seed set.
    """
    c = copy.deepcopy(cfg)
    for k in ("n_jobs", "output_dir"):
        c["experiment"].pop(k, None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
