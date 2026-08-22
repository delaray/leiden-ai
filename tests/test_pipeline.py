import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hnsw import LabelingConfig, PipelineConfig
from src.pipeline import run_pipeline


class DummyIndex:
    def __init__(self):
        self.k = None

    def knn_query(self, embeddings, k, num_threads):
        self.k = k
        n = len(embeddings)
        labels = np.arange(n * 2, dtype=np.int64).reshape(n, 2)
        labels = labels % 2
        distances = np.zeros((n, 2), dtype=np.float32)
        return labels, distances


def test_run_pipeline_returns_expected_structure(monkeypatch):
    events = []

    def fake_embed_sentences(sentences, config):
        return np.eye(len(sentences), 2, dtype=np.float32)

    def fake_build_hnsw_index(embeddings, config):
        return DummyIndex()

    def fake_build_knn_graph(embeddings, index, config, query_batch_size=10000):
        return {"kind": "graph", "nodes": len(embeddings)}

    def fake_hierarchical_leiden(graph, config):
        class DummyRoot:
            cluster_id = "root"
            depth = 0
            indices = np.array([0, 1], dtype=np.int64)
            children = []
            representative_indices = [0]
            resolution = config.resolution
        return DummyRoot()

    def fake_add_representatives(node, embeddings, n_representatives=5):
        node.representative_indices = [0]
        events.append("represented")

    def fake_label_cluster_tree(node, sentences, config):
        assert node.representative_indices == [0]
        node.label = "Test label"
        events.append("labeled")

    monkeypatch.setattr("src.pipeline.embed_sentences", fake_embed_sentences)
    monkeypatch.setattr("src.pipeline.build_hnsw_index", fake_build_hnsw_index)
    monkeypatch.setattr("src.pipeline.build_knn_graph", fake_build_knn_graph)
    monkeypatch.setattr("src.pipeline.hierarchical_leiden", fake_hierarchical_leiden)
    monkeypatch.setattr("src.pipeline.add_representatives", fake_add_representatives)
    monkeypatch.setattr("src.pipeline.label_cluster_tree", fake_label_cluster_tree)

    config = PipelineConfig(labeling=LabelingConfig(min_cluster_size=1))
    result = run_pipeline(["first", "second"], config)

    assert result["sentences"] == ["first", "second"]
    assert result["embeddings"].shape == (2, 2)
    assert result["graph"]["nodes"] == 2
    assert result["tree"].cluster_id == "root"
    assert result["tree"].label == "Test label"
    assert events == ["represented", "labeled"]


def test_pipeline_config_loads_labeling_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "labeling:\n  model: test-model\n  min_cluster_size: 3\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.labeling.model == "test-model"
    assert config.labeling.min_cluster_size == 3
