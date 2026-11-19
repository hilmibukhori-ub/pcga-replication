import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dapilot.config import load_config  # noqa: E402
from dapilot.dgp import generate_world  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs/pilot.yaml")


@pytest.fixture(scope="session")
def world(cfg):
    return generate_world(cfg, seed=12345)
