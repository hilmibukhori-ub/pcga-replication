"""Write PREREG_REV1_FREEZE.md (run once, on the laptop, after the dev rehearsal).

Hashes: the Revision-1 documents and code, the frozen parent code they call, the parent result files that
the analysis reuses (laptop: RSR real s0 and the 99 parent placebos per market), and a manifest of the raw
google_finance files used by the size measure. Hyperion-only parent outputs (RT-GCN real s0, perm000-018)
are hashed on hyperion with --remote-list and appended."""
import argparse, hashlib, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CODE = ["PREREG_REV1.md", "configs/rev1.yaml", "dapilot/rev1_sizebin.py", "dapilot/rev1_report.py",
        "scripts/run_rev1_rsr_sizebin.py", "scripts/rev1_verify_freeze.py", "scripts/hyperion/rev1_k99.sbatch",
        "scripts/hyperion/rev1_k99_tasks.txt", "tests/test_rev1.py",
        # frozen parent code called unchanged
        "dapilot/rsr_replica.py", "dapilot/rsr_replica_report.py", "configs/rsr_replica.yaml",
        "dapilot/rtgcn_replica.py", "dapilot/rtgcn_report.py", "scripts/run_rtgcn_replica.py", "configs/rtgcn_replica.yaml",
        "dapilot/rtgcn_ext.py", "dapilot/rtgcn_ext_report.py", "scripts/run_rtgcn_ext.py", "configs/rtgcn_ext.yaml",
        "data/raw/rsr_repo/data/NASDAQ_aver_line_dates.csv", "data/raw/rsr_repo/data/NYSE_aver_line_dates.csv"]
MANIFEST = "outputs/rev1_freeze_manifest_google_finance.txt"


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-list", help="file of '<sha256>  <path>' lines computed on hyperion")
    ap.add_argument("--blinding", required=True, help="blinding statement text")
    a = ap.parse_args()
    from dapilot import rsr_replica as rr
    lines = []
    for m in ("NASDAQ", "NYSE"):
        for t in rr.tickers(m):
            p = f"data/raw/rsr_repo/data/google_finance/{m}_{t}_30Y.csv"
            lines.append(f"{sha(p)}  {p}")
    (ROOT / MANIFEST).write_text("\n".join(lines) + "\n")
    files = CODE + [MANIFEST]
    for m in ("NASDAQ", "NYSE"):
        files.append(f"outputs/rsr_replica_v1/{m}_wikidata_real_s0.npz")
        files += [f"outputs/rsr_replica_v1/{m}_wikidata_perm{k:03d}_s0.npz" for k in range(99)]
    rows = [(p, sha(p)) for p in files]
    if a.remote_list:
        for ln in Path(a.remote_list).read_text().splitlines():
            h, p = ln.split()
            rows.append((p, h))
    out = [f"# Freeze record — PREREG_REV1.md (Revision 1 robustness)", "",
           f"**Frozen:** {datetime.now().strftime('%Y-%m-%d %H:%M')} CET.", "",
           f"**Blinding statement:** {a.blinding}", "",
           "| file | sha256 |", "|---|---|"] + [f"| `{p}` | `{h}` |" for p, h in rows]
    (ROOT / "PREREG_REV1_FREEZE.md").write_text("\n".join(out) + "\n")
    print(f"wrote PREREG_REV1_FREEZE.md with {len(rows)} rows")


if __name__ == "__main__":
    main()
