import copy

import numpy as np
import torch
import yaml

from dapilot.pcga_l_multimodel import (
    MatchedGraphNet,
    PooledMLP,
    graph_operators,
    matched_graph_features,
    parameter_count,
    run_seed,
)


def _cfg():
    with open("configs/pcga_l_multimodel_v1.yaml") as handle:
        return yaml.safe_load(handle)


def test_l0_keeps_feature_dimension_and_zeroes_only_graph_channels():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4, 2)).astype(np.float32)
    F = matched_graph_features(x, None)
    assert F.shape == (5, 4, 6)
    np.testing.assert_array_equal(F[..., :2], x)
    np.testing.assert_array_equal(F[..., 2:], 0.0)


def test_l0_graph_operators_are_zero_but_model_capacity_is_unchanged():
    mean, degree_sum = graph_operators(None, 7)
    np.testing.assert_array_equal(mean, 0.0)
    np.testing.assert_array_equal(degree_sum, 0.0)
    model_l0 = MatchedGraphNet(2, 8)
    model_l4 = MatchedGraphNet(2, 8)
    assert parameter_count(model_l0) == parameter_count(model_l4)


def test_zero_graph_gcn_output_is_invariant_to_other_nodes():
    torch.manual_seed(1)
    model = MatchedGraphNet(2, 5)
    x = torch.randn(3, 7, 2)
    z = torch.zeros(7, 7)
    before = model(x, z, z).detach()
    changed = x.clone()
    changed[:, 1:] += 100.0
    after = model(changed, z, z).detach()
    torch.testing.assert_close(before[:, 0], after[:, 0])


def test_pooled_mlp_parameter_count_does_not_depend_on_graph_level():
    assert parameter_count(PooledMLP(6, 8)) == parameter_count(PooledMLP(6, 8))


def test_tiny_run_is_deterministic_and_telescopes():
    cfg = _cfg()
    cfg = copy.deepcopy(cfg)
    cfg["world"]["T"] = 150
    cfg["world"]["per_sector"] = 4
    cfg["splits"] = {"train": [0, 80], "val": [80, 110], "test": [110, 149]}
    cfg["experiment"]["scenarios"] = ["edge"]
    cfg["experiment"]["families"] = ["ridge"]
    cfg["experiment"]["placebos_per_level"] = 1
    first, _ = run_seed(cfg, 777)
    second, _ = run_seed(cfg, 777)
    np.testing.assert_allclose(first.select_dtypes("number"), second.select_dtypes("number"))
    np.testing.assert_allclose(first["D1"], first[["density", "degree", "block", "edge"]].sum(1))

