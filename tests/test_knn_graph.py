import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import knn_graph
from src.hnsw import HNSWConfig


class FakeIndex:
    def __init__(self):
        self.calls = []

    def knn_query(self, embeddings, k, num_threads):
        self.calls.append((embeddings.shape, k, num_threads))
        labels = np.array([[0, 1], [1, 0]], dtype=np.int64)
        distances = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)
        return labels, distances


def test_build_knn_graph_creates_weighted_igraph(monkeypatch):
    fake_index = FakeIndex()
    config = HNSWConfig(k=1, min_similarity=0.1, num_threads=2)

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    graph = knn_graph.build_knn_graph(embeddings, fake_index, config, query_batch_size=1)

    assert graph.vcount() == 2
    assert graph.ecount() >= 1
    assert all(weight >= 0.0 for weight in graph.es["weight"])
