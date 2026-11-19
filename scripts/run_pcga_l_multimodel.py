"""Run, freeze, and summarise the prospective PCGA-L cross-model experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.pcga_l import STEPS, recovery_summary  # noqa: E402
from dapilot.pcga_l_multimodel import run_seed, write_seed_result  # noqa: E402
from dapilot.stats import holm, paired_summary, verdict  # noqa: E402


FROZEN_FILES = [
    "PREREG_PCGA_L_MULTIMODEL.md",
    "configs/pcga_l_multimodel_v1.yaml",
    "dapilot/pcga_l.py",
    "dapilot/pcga_l_multimodel.py",
    "scripts/run_pcga_l_multimodel.py",
    "tests/test_pcga_l.py",
    "tests/test_pcga_l_multimodel.py",
]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path):
    return yaml.safe_load(path.read_text())


def freeze(cfg: dict):
    manifest_path = ROOT / cfg["experiment"]["freeze_manifest"]
    if manifest_path.exists():
        raise FileExistsError(f"freeze already exists: {manifest_path}")
    manifest = {
        "experiment": cfg["experiment"]["name"],
        "created_before_confirmatory_run": True,
        "files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def verify_freeze(cfg: dict):
    path = ROOT / cfg["experiment"]["freeze_manifest"]
    if not path.exists():
        raise RuntimeError("confirmatory run blocked: freeze manifest does not exist")
    manifest = json.loads(path.read_text())
    changed = [name for name, digest in manifest["files"].items() if sha256(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f"confirmatory run blocked: frozen files changed: {changed}")


def _output_dir(cfg, mode):
    key = "development_output_dir" if mode == "dev" else "confirmatory_output_dir"
    return ROOT / cfg["experiment"][key]


def run_one(cfg: dict, mode: str, seed: int):
    allowed = cfg["experiment"]["development_seeds" if mode == "dev" else "confirmatory_seeds"]
    if seed not in allowed:
        raise ValueError(f"seed {seed} is not a pre-specified {mode} seed")
    if mode == "confirm":
        verify_freeze(cfg)
    out = _output_dir(cfg, mode)
    result_path = out / f"seed_{seed:05d}.csv"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed seed: {result_path}")
    rows, diagnostics = run_seed(cfg, seed)
    write_seed_result(rows, diagnostics, out, seed)
    print(f"completed {mode} seed {seed}", flush=True)


def _worker(cfg: dict, mode: str, seed: int):
    torch.set_num_threads(1)
    run_one(cfg, mode, seed)
    return seed


def run_many(cfg: dict, mode: str, jobs: int):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    key = "development_seeds" if mode == "dev" else "confirmatory_seeds"
    out = _output_dir(cfg, mode)
    seeds = [
        seed for seed in cfg["experiment"][key]
        if not (out / f"seed_{seed:05d}.csv").exists()
    ]
    if mode == "confirm":
        verify_freeze(cfg)
    if not seeds:
        print(f"all {mode} seeds already complete")
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_worker, cfg, mode, seed) for seed in seeds]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _load_confirmatory(cfg):
    out = _output_dir(cfg, "confirm")
    expected = set(cfg["experiment"]["confirmatory_seeds"])
    found = {int(path.stem.split("_")[1]) for path in out.glob("seed_*.csv") if "diagnostics" not in path.stem}
    if found != expected:
        raise RuntimeError(f"confirmatory results incomplete: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    return pd.concat(
        [pd.read_csv(out / f"seed_{seed:05d}.csv", keep_default_na=False) for seed in sorted(expected)],
        ignore_index=True,
    )


def summarise(cfg):
    verify_freeze(cfg)
    out = _output_dir(cfg, "confirm")
    frame = _load_confirmatory(cfg)
    frame.to_csv(out / "per_seed.csv", index=False)
    hcfg = cfg["hypotheses"]
    expected = hcfg["expected_steps"]

    recovery_frames = []
    for family, group in frame.groupby("family", sort=False):
        summary = recovery_summary(
            group, expected_steps=expected,
            share_threshold=float(hcfg["attribution_share"]),
        )
        summary.insert(0, "family", family)
        summary["concentration_met"] = (
            summary["recovered"]
            & (summary["fraction_share_at_least_threshold"] >= float(hcfg["attribution_seed_frac"]))
        )
        recovery_frames.append(summary)
    recovery = pd.concat(recovery_frames, ignore_index=True)
    recovery.to_csv(out / "recovery_summary.csv", index=False)

    specifications = []
    for family in cfg["experiment"]["families"]:
        for scenario, step in expected.items():
            specifications.append((f"signal_{family}_{scenario}", family, scenario, step, "positive"))
        for step in STEPS:
            specifications.append((f"null_{family}_{step}", family, "null", step, "two-sided"))
    summaries = []
    for _, family, scenario, step, _ in specifications:
        values = frame[(frame.family == family) & (frame.scenario == scenario)].sort_values("seed")[step]
        summaries.append(paired_summary(values.to_numpy()))
    adjusted = holm([item["wilcoxon_p"] for item in summaries])
    rows = []
    for spec, result, p_adj in zip(specifications, summaries, adjusted):
        identifier, family, scenario, step, predicted = spec
        rows.append({
            "id": identifier, "family": family, "scenario": scenario, "step": step,
            "predicted": predicted, **result, "p_holm": p_adj,
            "verdict": verdict(result, p_adj, predicted, float(hcfg["alpha"])),
        })
    hypotheses = pd.DataFrame(rows)
    hypotheses.to_csv(out / "hypotheses.csv", index=False)

    signal = hypotheses[hypotheses.id.str.startswith("signal_")]
    concentration = recovery[recovery.expected_step != "none"]
    global_met = bool((signal.verdict == "supported").all() and concentration.concentration_met.all())
    lines = [
        "# PCGA-L cross-model construct recovery",
        "",
        "Prospective protocol: `PREREG_PCGA_L_MULTIMODEL.md`. Confirmatory DGP seeds 100--119; ",
        "all graph variants retrained; positive values mean that the added structure lowers test loss.",
        "",
        f"**Global pre-specified recovery criterion: {'MET' if global_met else 'NOT MET'}.**",
        "",
        "## Signal-scenario recovery",
        "",
        "| family | scenario | expected | mean step | expected share | seeds >= 70% | p Holm | directional test | concentration |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for test in signal.itertuples():
        rec = concentration[(concentration.family == test.family) & (concentration.scenario == test.scenario)].iloc[0]
        lines.append(
            f"| {test.family} | {test.scenario} | {test.step} | {test.mean:+.3e} | "
            f"{rec.mean_expected_share:.3f} | {rec.fraction_share_at_least_threshold:.0%} | "
            f"{test.p_holm:.3g} | {test.verdict} | {'met' if rec.concentration_met else 'not met'} |"
        )
    lines.extend([
        "",
        "## Null negative controls",
        "",
        "| family | step | mean | p Holm | verdict |",
        "|---|---|---:|---:|---|",
    ])
    for test in hypotheses[hypotheses.id.str.startswith("null_")].itertuples():
        lines.append(f"| {test.family} | {test.step} | {test.mean:+.3e} | {test.p_holm:.3g} | {test.verdict} |")
    lines.extend([
        "",
        "Null non-rejection is not equivalence. This experiment establishes construct recovery only ",
        "for the fixed synthetic worlds and matched experts; it is not market or causal evidence.",
        "",
    ])
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps({"global_recovery_met": global_met}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pcga_l_multimodel_v1.yaml")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--mode", choices=["dev", "confirm"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true", help="run all incomplete seeds for --mode")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(1)
    if args.freeze:
        print(freeze(config))
    elif args.summarise:
        summarise(config)
    elif args.mode is not None and args.all:
        run_many(config, args.mode, args.jobs)
    elif args.mode is not None and args.seed is not None:
        run_one(config, args.mode, args.seed)
    else:
        parser.error("choose --freeze, --summarise, or both --mode and --seed")
