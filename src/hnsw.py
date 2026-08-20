from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json

import hnswlib
import igraph as ig
import leidenalg as la
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


# ============================================================
# Configuration
# ============================================================

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 256
    device: str = "cuda"
    normalize: bool = True


@dataclass
class HNSWConfig:
    k: int = 30

    # HNSW construction parameters
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 100

    # Edges below this cosine similarity are discarded.
    min_similarity: float = 0.25

    num_threads: int = -1


@dataclass
class LeidenConfig:
    # Initial Leiden resolution
    resolution: float = 0.5

    # Multiply resolution by this at each hierarchy level.
    resolution_multiplier: float = 1.5

    max_depth: int = 5

    # Don't recursively partition clusters below this size.
    min_cluster_size: int = 50

    # Children smaller than this can optionally be regarded
    # as too small to be useful semantic topics.
    min_child_size: int = 10

    n_iterations: int = 4

    seed: int = 42


@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    leiden: LeidenConfig = field(default_factory=LeidenConfig)


# ============================================================
# Hierarchy data structure
# ============================================================

@dataclass
class ClusterNode:
    cluster_id: str
    depth: int
    indices: np.ndarray
    resolution: float

    children: list["ClusterNode"] = field(default_factory=list)

    # Useful descriptive information
    representative_indices: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ============================================================
# Embeddings
# ============================================================

def embed_sentences(
    sentences: list[str],
    config: EmbeddingConfig,
) -> np.ndarray:
    """
    Embed all sentences and return an N x D float32 NumPy matrix.
    """

    print(f"Loading embedding model: {config.model_name}")

    model = SentenceTransformer(
        config.model_name,
        device=config.device,
    )

    print(f"Embedding {len(sentences):,} sentences...")

    embeddings = model.encode(
        sentences,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"Embeddings: {embeddings.shape[0]:,} × "
        f"{embeddings.shape[1]}"
    )

    return embeddings


# ============================================================
# HNSW
# ============================================================

def build_hnsw_index(
    embeddings: np.ndarray,
    config: HNSWConfig,
) -> hnswlib.Index:
    """
    Construct an HNSW cosine index.
    """

    n, dim = embeddings.shape

    index = hnswlib.Index(
        space="cosine",
        dim=dim,
    )

    index.init_index(
        max_elements=n,
        ef_construction=config.ef_construction,
        M=config.M,
    )

    index.set_num_threads(config.num_threads)

    ids = np.arange(n)

    print("Building HNSW index...")

    index.add_items(
        embeddings,
        ids,
    )

    index.set_ef(
        max(config.ef_search, config.k + 1)
    )

    return index

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
