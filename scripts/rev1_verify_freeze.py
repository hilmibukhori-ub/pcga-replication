"""Exit non-zero unless every file listed in PREREG_REV1_FREEZE.md that exists here matches its sha256.
Files absent on this machine (laptop-only or hyperion-only) are reported, not failed."""
import hashlib, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "PREREG_REV1_FREEZE.md"
if not F.exists():
    sys.exit("PREREG_REV1_FREEZE.md missing (not frozen)")
rows = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", F.read_text())
if not rows:
    sys.exit("no hashes in freeze file")
absent = [p for p, _ in rows if not (ROOT / p).exists()]
bad = [p for p, h in rows if (ROOT / p).exists() and hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
if bad:
    sys.exit(f"FREEZE MISMATCH: {bad}")
print(f"PREREG_REV1_FREEZE ok: {len(rows) - len(absent)} verified, {len(absent)} absent here")
