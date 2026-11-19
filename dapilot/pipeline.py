"""Single-seed pipeline: data -> frozen experts -> selector features -> selectors (fit on
pre-test data only) -> locked-test evaluation. Writes predictions, outputs, metrics, logs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import config_hash, save_config
from .dgp import REGIME_NAMES, generate_world
from .experts import label_array, train_all_experts
from .features import build_selector_features, make_meta_data
from .graphs import (STRUCTURAL_NAMES, consensus_graph, permute_graph,
                     structural_diagnostics)
from .metrics import cond_risk, evaluate_graph_value, evaluate_selectors
from .selectors import build_selectors
from .splits import SPLIT_ORDER, TemporalSplits


class RunLog:
    def __init__(self, directory: Path):
        self.txt = open(directory / "log.txt", "w")
        self.jsonl = open(directory / "events.jsonl", "w")
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        self.txt.write(line + "\n")
        self.txt.flush()

    def event(self, kind: str, **payload):
        rec = {"t": round(time.time() - self.t0, 2), "event": kind, **payload}
        self.jsonl.write(json.dumps(rec, default=_json_default) + "\n")
        self.jsonl.flush()

    def close(self):
        self.txt.close()
        self.jsonl.close()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(o):
    """Recursively convert to JSON-safe values (NaN -> None)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def slot_structural_diagnostics(obs, splits, adjacencies: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build diagnostics from the exact adjacency consumed by each graph-expert slot."""
    tr = splits.decision_times("expert_train")
    if tr.max() >= splits.start("meta_train"):
        raise ValueError("structural consensus must be fixed before meta training")
    out = {}
    for slot, adjacency in adjacencies.items():
        consensus = consensus_graph(adjacency[tr])
        out[slot] = structural_diagnostics(
            adjacency, obs.X, obs.sectors, consensus
        )
    return out


def run_slot(slot, obs, preds, g_key, n_key, placebo_keys, struct, splits, cfg, seed, log):
    feats = build_selector_features(obs, preds, g_key, n_key, placebo_keys, struct,
                                    STRUCTURAL_NAMES, splits, cfg["selectors"])
    md = make_meta_data(obs, preds, g_key, n_key, placebo_keys, feats, splits)
    outputs, hp = {}, {}
    for sel in build_selectors(cfg):
        t0 = time.time()
        hp[sel.name] = sel.fit(md, seed, cfg)
        outputs[sel.name] = {"w": sel.weight(md), "benefit": sel.benefit(md), "d2": sel.d2(md)}
        log(f"[{slot}] selector {sel.name} fitted in {time.time() - t0:.1f}s")
    return feats, md, outputs, hp


def run_seed(cfg: dict, seed: int, run_dir: str | Path) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(run_dir) / f"seed_{seed:02d}"
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    log = RunLog(out)
    log.event("start", seed=seed, config_hash=config_hash(cfg))

    world = generate_world(cfg, seed)
    obs, hid = world.observable, world.hidden
    splits = TemporalSplits(cfg["splits"], obs.T)
    h = splits.horizon
    coverage = {n: np.bincount(hid.regime[splits.decision_times(n)], minlength=3).tolist()
                for n in SPLIT_ORDER}
    log.event("data", splits=splits.describe(), regime_counts_per_split=coverage,
              true_edges=int(hid.A_true.sum() // 2))
    log(f"data generated; regime coverage per split {coverage}")

    # --- experts (frozen after this block)
    preds, expert_info, perms = train_all_experts(obs, splits, cfg, seed, log)
    log.event("experts", info=expert_info)
    v0 = splits.start("expert_val")
    np.savez_compressed(
        out / "predictions.npz",
        decision_time_0based=np.arange(v0, obs.T - h),
        **{k: p[v0:obs.T - h] for k, p in preds.items()},
        placebo_perms=np.stack(perms),
    )

    # --- selectors: real-graph slot and placebo-graph slot
    scfg = cfg["selectors"]
    placebo = [k for k in preds if k.startswith("placebo_")]
    if not placebo:
        raise ValueError("at least one placebo expert is required for the placebo slot")
    slots = {
        "real": (scfg["graph_expert"], scfg["nongraph_expert"], placebo),
        "placebo": (placebo[0], scfg["nongraph_expert"], placebo[1:]),
    }
    slot_struct = slot_structural_diagnostics(
        obs,
        splits,
        {
            "real": obs.A_obs,
            "placebo": permute_graph(obs.A_obs, perms[0]),
        },
    )
    fitted = {}
    for slot, (g, n, pk) in slots.items():
        feats, md, outputs, hp = run_slot(
            slot, obs, preds, g, n, pk, slot_struct[slot], splits, cfg, seed, log
        )
        fitted[slot] = (feats, md, outputs)
        log.event("selectors_fitted", slot=slot, hyperparams=hp)
        with open(out / f"hyperparams_{slot}.json", "w") as f:
            json.dump(_clean(hp), f, indent=1)

    # ===================== LOCKED TEST: everything above is frozen ======================
    log.event("locked_test_open")
    test_idx = splits.decision_times("test")
    y = label_array(obs, h).astype(np.float64)
    metrics = {"seed": seed, "config_hash": config_hash(cfg), "regime_counts_test":
               dict(zip(REGIME_NAMES, np.bincount(hid.regime[test_idx], minlength=3).tolist()))}
    metrics["graph_value"] = evaluate_graph_value(
        preds, y, hid.mu, hid.regime, test_idx, graph=scfg["graph_expert"], placebos=placebo)
    metrics["selectors"] = {}
    sel_npz = {}
    for slot, (g, n, pk) in slots.items():
        feats, md, outputs = fitted[slot]
        RP = np.mean([cond_risk(hid.mu, preds[k]) for k in pk], 0) if pk else None
        metrics["selectors"][slot] = evaluate_selectors(
            outputs, preds[g], preds[n], y, hid.mu, hid.regime, test_idx,
            cfg["evaluation"]["calibration_bins"], RP=RP)
        for name, o in outputs.items():
            for k in ("w", "benefit", "d2"):
                if o[k] is not None:
                    sel_npz[f"{slot}/{name}/{k}"] = o[k][v0:obs.T - h].astype(np.float32)
        if slot == "real":
            np.savez_compressed(out / "selector_features.npz", Z=feats.Z[v0:obs.T - h],
                                names=np.array(feats.names))
    np.savez_compressed(out / "selector_outputs.npz",
                        decision_time_0based=np.arange(v0, obs.T - h), **sel_npz)

    # evaluation-only hidden quantities for plots (never inputs)
    RG = cond_risk(hid.mu[:-h], preds[scfg["graph_expert"]][:-h])
    RN = cond_risk(hid.mu[:-h], preds[scfg["nongraph_expert"]][:-h])
    np.savez_compressed(out / "hidden_eval.npz", regime=hid.regime[v0:obs.T - h],
                        corruption=hid.corruption[v0:obs.T - h],
                        benefit_star=(RN - RG)[v0:], decision_time_0based=np.arange(v0, obs.T - h))
    with open(out / "metrics.json", "w") as f:
        json.dump(_clean(metrics), f, indent=1)
    log.event("done")
    log("done")
    log.close()
    return metrics
