"""Summarise the RT-GCN re-test (PREREG_RTGCN.md): python scripts/rtgcn_summarise.py"""
import sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dapilot import rtgcn_report as rep  # noqa: E402
rep.summarise(yaml.safe_load((ROOT / "configs/rtgcn_replica.yaml").read_text()))
