import copy
import json

import numpy as np
import pytest

from dapilot.config import config_hash
from dapilot.graphs import permute_graph
from dapilot.pipeline import slot_structural_diagnostics
from dapilot.report import load_seeds
from dapilot.splits import TemporalSplits


def test_slot_diagnostics_use_the_experts_actual_adjacency(cfg, world):
    obs = world.observable
    splits = TemporalSplits(cfg["splits"], obs.T)
    perm = np.random.default_rng(91).permutation(obs.N)
    placebo_adjacency = permute_graph(obs.A_obs, perm)
    got = slot_structural_diagnostics(
        obs, splits, {"real": obs.A_obs, "placebo": placebo_adjacency}
    )

    real_only = slot_structural_diagnostics(obs, splits, {"x": obs.A_obs})["x"]
    placebo_only = slot_structural_diagnostics(obs, splits, {"x": placebo_adjacency})["x"]
    np.testing.assert_array_equal(got["real"], real_only)
    np.testing.assert_array_equal(got["placebo"], placebo_only)
    # Sector alignment and feature homophily should change under a node permutation.
    assert not np.allclose(got["real"][:, [1, 5]], got["placebo"][:, [1, 5]])


def _write_metric(root, seed, digest):
    directory = root / f"seed_{seed:02d}"
    directory.mkdir()
    (directory / "metrics.json").write_text(
        json.dumps({"seed": seed, "config_hash": digest})
    )


def test_seed_loader_rejects_missing_extra_or_mixed_results(tmp_path, cfg):
    run_cfg = copy.deepcopy(cfg)
    run_cfg["experiment"]["seeds"] = [0, 1]
    digest = config_hash(run_cfg)
    assert digest != config_hash(cfg)

    _write_metric(tmp_path, 0, digest)
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        load_seeds(tmp_path, run_cfg)

    _write_metric(tmp_path, 1, digest)
    assert [m["seed"] for m in load_seeds(tmp_path, run_cfg)] == [0, 1]

    _write_metric(tmp_path, 2, digest)
    with pytest.raises(ValueError, match=r"extra=\[2\]"):
        load_seeds(tmp_path, run_cfg)

    (tmp_path / "seed_02" / "metrics.json").unlink()
    (tmp_path / "seed_02").rmdir()
    (tmp_path / "seed_01" / "metrics.json").write_text(
        json.dumps({"seed": 1, "config_hash": "wrong"})
    )
    with pytest.raises(ValueError, match="config hash"):
        load_seeds(tmp_path, run_cfg)
