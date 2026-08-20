import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import leiden
from src.hnsw import ClusterNode, LeidenConfig


class FakeGraph:
    def __init__(self, vertex_count=4):
        self._vertex_count = vertex_count

    def vcount(self):
        return self._vertex_count

    def ecount(self):
        return 2

    def induced_subgraph(self, indices):
        return FakeGraph(vertex_count=len(indices))


def test_leiden_partition_returns_community_lists(monkeypatch):
    fake_graph = FakeGraph(vertex_count=4)

    def fake_find_partition(graph, partition_type, weights, resolution_parameter, n_iterations, seed):
        assert weights == "weight"
        assert resolution_parameter == 0.7
        assert n_iterations == 5
        assert seed == 42
        return [np.array([0, 1]), np.array([2, 3])]

    monkeypatch.setattr(leiden.la, "find_partition", fake_find_partition)

    communities = leiden.leiden_partition(fake_graph, resolution=0.7, config=LeidenConfig(n_iterations=5, seed=42))

    assert communities == [[0, 1], [2, 3]]


def test_hierarchical_leiden_builds_root_and_children(monkeypatch):
    graph = FakeGraph(vertex_count=8)
    calls = []

    def fake_partition(subgraph, resolution, config):
        calls.append(resolution)
        if len(calls) == 1:
            return [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7])]
        return [np.array([0, 1]), np.array([2])]

    monkeypatch.setattr(leiden, "leiden_partition", fake_partition)

    root = leiden.hierarchical_leiden(graph, LeidenConfig(min_cluster_size=2, min_child_size=2, max_depth=2))

    assert root.cluster_id == "root"
    assert root.depth == 0
    assert len(root.children) == 2
    assert root.children[0].cluster_id.startswith("root.")
    assert all(isinstance(child, ClusterNode) for child in root.children)
