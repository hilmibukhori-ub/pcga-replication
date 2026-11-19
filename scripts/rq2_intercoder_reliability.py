#!/usr/bin/env python3
"""Compute pre-reconciliation agreement for the 36-paper RQ2 audit.

Coder 1 used descriptive text in the original sheet.  The normalisation below
maps that text to the categories in the frozen codebook used by coder 2.  The
two original sheets are read-only inputs and are never modified.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODER1 = ROOT / "literature" / "syssearch" / "rq2_coding.csv"
CODER2 = ROOT / "literature" / "03_CODING_SHEET_syaiful_CLEAN.csv"

FIELDS = (
    "no_relation_baseline",
    "same_arch_no_graph",
    "random_or_placebo_graph",
    "n_runs",
    "sig_test",
    "validation_split_for_selection",
    "code_released",
)


def canonical_doi(value: str) -> str:
    return value.strip().lower().removeprefix("https://doi.org/")


def normalise_coder1(field: str, value: str) -> str:
    value = (value or "").strip().lower()

    if field == "no_relation_baseline":
        return "yes" if value.startswith("yes") else "no"

    if field == "same_arch_no_graph":
        if value.startswith("yes"):
            return "clean"
        if value.startswith("partial"):
            return "partial"
        return "no"

    if field == "random_or_placebo_graph":
        if "partial" in value:
            return "partial"
        return "yes" if value.startswith("yes") else "no"

    if field == "n_runs":
        explicit_multiple = re.search(
            r"(^|\b)(5|10|five|ten|several|multiple|repeated)(\b|,)", value
        )
        return ">1" if explicit_multiple and "not reported" not in value else "not stated"

    if field == "sig_test":
        negative = (
            not value
            or value.startswith("none")
            or value.startswith("no")
            or "correlation test only" in value
        )
        return "no" if negative else "yes"

    if field == "validation_split_for_selection":
        if value.startswith("no:") or "test-set" in value or "test set" in value:
            return "test set"
        if value.startswith("yes"):
            return "yes"
        # The two legacy 'partial' entries do not establish that validation was
        # used for selection, so the frozen codebook maps them to not reported.
        return "not reported"

    if field == "code_released":
        # The frozen codebook requires identifiable implementation code; data
        # alone and a promise of future release therefore map to no.
        return "yes" if value.startswith("yes") else "no"

    raise ValueError(f"Unknown field: {field}")


def kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires two non-empty lists of equal length")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    labels = set(left) | set(right)
    expected = sum(left.count(x) * right.count(x) for x in labels) / (n * n)
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0


def main() -> None:
    with CODER1.open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    with CODER2.open(encoding="utf-8-sig", newline="") as handle:
        second = list(csv.DictReader(handle))

    coder1_by_doi = {canonical_doi(row["doi"]): row for row in original}
    if len(second) != 36:
        raise SystemExit(f"Expected 36 coder-2 rows, found {len(second)}")

    joined = []
    for row2 in second:
        doi = canonical_doi(row2["doi"])
        if doi not in coder1_by_doi:
            raise SystemExit(f"Coder-2 DOI not found in coder-1 sheet: {doi}")
        joined.append((row2["paper_id"], coder1_by_doi[doi], row2))

    print("| Item | Agreement | Cohen's kappa | Disagreements |")
    print("|---|---:|---:|---:|")
    total_agree = 0
    total_cells = len(joined) * len(FIELDS)
    disagreement_ids: dict[str, list[str]] = {}

    for field in FIELDS:
        left = [normalise_coder1(field, row1[field]) for _, row1, _ in joined]
        right = [row2[field].strip().lower() for _, _, row2 in joined]
        invalid = sorted(set(right) - set(left) - {
            "yes", "no", "clean", "partial", "1", ">1", "not stated",
            "not reported", "test set",
        })
        if invalid:
            raise SystemExit(f"Unexpected coder-2 values for {field}: {invalid}")

        agreed = sum(a == b for a, b in zip(left, right))
        ids = [paper_id for (paper_id, _, _), a, b in zip(joined, left, right) if a != b]
        disagreement_ids[field] = ids
        total_agree += agreed
        print(f"| `{field}` | {agreed}/36 ({agreed / 36:.1%}) | {kappa(left, right):.3f} | {36 - agreed} |")

    print(f"\nOverall raw agreement: {total_agree}/{total_cells} ({total_agree / total_cells:.1%}).")
    print(f"Disputed cells before reconciliation: {total_cells - total_agree}.")
    print("\nDisagreement paper IDs:")
    for field, ids in disagreement_ids.items():
        print(f"- `{field}`: {', '.join(ids) if ids else 'none'}")


if __name__ == "__main__":
    main()
