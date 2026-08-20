import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hnsw


class FakeSentenceTransformer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def encode(self, sentences, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        assert sentences == ["a", "b"]
        assert batch_size == 16
        assert convert_to_numpy is True
        assert normalize_embeddings is True
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class FakeHNSWIndex:
    def __init__(self, space, dim):
        self.space = space
        self.dim = dim
        self.ef = None
        self.num_threads = None
        self.max_elements = None

    def init_index(self, max_elements, ef_construction, M):
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M

    def set_num_threads(self, num_threads):
        self.num_threads = num_threads

    def add_items(self, embeddings, ids):
        self.embeddings = embeddings
        self.ids = ids

    def set_ef(self, ef):
        self.ef = ef


def test_embed_sentences_returns_float32_matrix(monkeypatch):
    monkeypatch.setattr(hnsw, "SentenceTransformer", FakeSentenceTransformer)

    config = hnsw.EmbeddingConfig(batch_size=16, normalize=True)
    vectors = hnsw.embed_sentences(["a", "b"], config)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32


def test_build_hnsw_index_initializes_index(monkeypatch):
    monkeypatch.setattr(hnsw.hnswlib, "Index", FakeHNSWIndex)

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    config = hnsw.HNSWConfig(k=2, M=12, ef_construction=40, ef_search=10, num_threads=3)

    index = hnsw.build_hnsw_index(embeddings, config)

    assert isinstance(index, FakeHNSWIndex)
    assert index.space == "cosine"
    assert index.dim == 2
    assert index.max_elements == 2
    assert index.num_threads == 3
    assert index.ef == max(10, 2 + 1)


def test_embedding_default_device_falls_back_to_cpu_without_cuda(monkeypatch):
    calls = {}

    class FakeTorch:
        @staticmethod
        def cuda_is_available():
            return False

    monkeypatch.setattr(hnsw, "torch", FakeTorch)

    config = hnsw.EmbeddingConfig()

    assert config.device == "cpu"
