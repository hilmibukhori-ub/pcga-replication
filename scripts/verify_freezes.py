"""Check every SHA-256 freeze record in this repository against the files that are present.

Run from the repository root:  python scripts/verify_freezes.py

Each freeze record lists `path` -> sha256 for the files that were frozen before a confirmatory run.
Files that are not shipped in this repository (raw data, upstream code, large prediction arrays) are
reported as "absent"; every file that is present must match its recorded hash. A superseded record
(PREREG_RTGCN_FREEZE_v0.1_superseded.md) is reported but not required to match.
"""
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def records():
    for f in sorted(ROOT.glob("*FREEZE*.md")):
        rows = [m.groups() for m in map(ROW.match, f.read_text().splitlines()) if m]
        if rows:
            yield f.name, rows
    j = ROOT / "PCGA_L_MULTIMODEL_FREEZE.json"
    if j.exists():
        rows = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) and "/" in k or k.endswith(".md"):
                        if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
                            rows.append((k, v))
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(json.loads(j.read_text()))
        yield j.name, rows


failed = False
for name, rows in records():
    ok = bad = absent = 0
    mism = []
    for path, h in rows:
        p = ROOT / path
        if not p.is_file():
            absent += 1
        elif sha(p) == h:
            ok += 1
        else:
            bad += 1
            mism.append(path)
    superseded = "superseded" in name
    tag = "MISMATCH" if bad and not superseded else ("superseded" if superseded else "ok")
    print(f"{name:42s} {tag:10s} verified {ok:3d}  mismatched {bad:2d}  absent {absent:3d}")
    for m in mism:
        print(f"    mismatch: {m}")
    failed |= bool(bad) and not superseded
sys.exit(1 if failed else 0)
