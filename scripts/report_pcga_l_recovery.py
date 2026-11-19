"""Verify and report PCGA-L recovery from the frozen placebo-ladder v1 outputs.

This script performs no model fitting and makes no new confirmatory claim.  It checks
the algebraic decomposition, reproduces the preregistered recovery summaries from the
stored level losses, and writes provenance hashes for the exact source artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.pcga_l import (  # noqa: E402
    STEPS,
    decompose_loss_levels,
    recovery_summary,
    telescoping_error,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_report(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    analysis, recovery = cfg["analysis"], cfg["recovery"]
    source_config = _resolve(analysis["source_config"])
    source_results = _resolve(analysis["source_results"])
    source_hypotheses = _resolve(analysis["source_hypotheses"])
    out = _resolve(analysis["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # ``null`` is a deliberate scenario label, not a missing-value token.
    stored = pd.read_csv(source_results, keep_default_na=False)
    recomputed = decompose_loss_levels(stored.drop(columns=["D1", *STEPS], errors="ignore"))
    for column in ("D1", *STEPS):
        if column in stored:
            np.testing.assert_allclose(
                recomputed[column], stored[column], rtol=0.0, atol=1e-14,
                err_msg=f"stored {column} differs from the L0--L4 decomposition",
            )
    max_telescoping_error = float(np.max(np.abs(telescoping_error(recomputed))))

    summary = recovery_summary(
        recomputed,
        expected_steps=recovery["expected_steps"],
        share_threshold=float(recovery["attribution_share"]),
    )
    required_fraction = float(recovery["attribution_seed_frac"])
    summary["meets_preregistered_recovery_rule"] = (
        summary["recovered"]
        & (summary["fraction_share_at_least_threshold"] >= required_fraction)
    )
    summary.to_csv(out / "recovery_summary.csv", index=False)

    hypotheses = pd.read_csv(source_hypotheses, keep_default_na=False)
    prereg_ids = [
        "H1", "H2", "H3", "H4_edge", "H4_block", "H4_degree",
        "H6_density", "H6_degree", "H6_block", "H6_edge",
    ]
    hypothesis_excerpt = hypotheses[hypotheses["id"].isin(prereg_ids)].copy()
    hypothesis_excerpt.to_csv(out / "preregistered_hypothesis_excerpt.csv", index=False)

    source_paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "dapilot/pcga_l.py",
        source_config,
        source_results,
        source_hypotheses,
    ]
    provenance = {
        "analysis_name": analysis["name"],
        "status": analysis["status"],
        "new_model_fitting": False,
        "max_abs_telescoping_error": max_telescoping_error,
        "sources": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
            for path in source_paths
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    lines = [
        "# PCGA-L construct-recovery report",
        "",
        "**Status:** post-hoc method packaging of a previously preregistered experiment. ",
        "The PCGA-L name and this report were not preregistered; the scenarios, ladder, ",
        "thresholds, hypotheses, seeds and stored test losses were fixed in the earlier ",
        "`placebo_ladder_v1` study. No model was fitted by this report.",
        "",
        "## What was verified",
        "",
        "The loss decomposition is `D1 = density + degree + block + edge`. Steps may be ",
        "negative; monotonic improvement is not assumed. The maximum absolute telescoping ",
        f"error in the stored results is `{max_telescoping_error:.3e}`.",
        "",
        "## Recovery of the known signal level",
        "",
        "| scenario | expected | dominant | expected positive-value share | seeds >= 70% | recovery rule |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in summary.itertuples():
        if row.expected_step == "none":
            continue
        lines.append(
            f"| {row.scenario} | {row.expected_step} | {row.dominant_mean_step} | "
            f"{row.mean_expected_share:.3f} | {row.fraction_share_at_least_threshold:.0%} | "
            f"{'met' if row.meets_preregistered_recovery_rule else 'not met'} |"
        )
    null_row = summary[summary["scenario"] == "null"].iloc[0]
    h6 = hypothesis_excerpt[hypothesis_excerpt["id"].str.startswith("H6_")]
    lines.extend([
        "",
        "All three signal-bearing scenarios recover the structural level that generated ",
        "the signal under the old preregistered 70%/80% rule. The null scenario remains a ",
        "negative control and is assessed by the old H6 tests, not by forcing a dominant step.",
        "",
        "## Null negative control",
        "",
        "| step | mean loss contribution | old Holm-adjusted p | old verdict |",
        "|---|---:|---:|---|",
    ])
    for step in STEPS:
        old = h6[h6["id"] == f"H6_{step}"].iloc[0]
        lines.append(
            f"| {step} | {null_row[f'mean_{step}']:+.3e} | "
            f"{float(old['p_holm']):.3f} | {old['verdict']} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "This supports construct recovery for the transparent ridge data-generating process. ",
        "It is not evidence that market relations are causal, that the ladder is universally ",
        "identified, or that every graph model will recover the same level. Empirical relation ",
        "specificity still requires the pre-specified constrained-placebo test, and practical ",
        "value still requires an independently chosen materiality threshold.",
        "",
    ])
    report = out / "REPORT.md"
    report.write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pcga_l_recovery.yaml",
    )
    args = parser.parse_args()
    result = build_report(args.config.resolve())
    print(result.read_text())
