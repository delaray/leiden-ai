"""Embedding, HNSW indexing, and cluster-tree configuration for hierarchical
semantic clustering.

This module turns raw sentences into dense vector embeddings, builds a
cosine-similarity nearest-neighbor index with HNSW, and defines the
configuration objects used throughout the pipeline. The HNSW step approximates
k-nearest neighbors efficiently in high-dimensional space, which is then
converted into a graph for graph-based clustering.

Classes
-------
EmbeddingConfig
    Settings for the sentence embedding model and output normalization.

HNSWConfig
    Parameters controlling the approximate nearest-neighbor index.

LeidenConfig
    Settings for the Leiden community-detection algorithm and hierarchy depth.

PipelineConfig
    Convenience container for the embedding, HNSW, and Leiden configuration
    blocks.

ClusterNode
    A node in the hierarchical cluster tree containing a set of sentence
    indices and children.

Functions
---------
embed_sentences
    Encodes a list of sentences into a normalized float32 embedding matrix.

build_hnsw_index
    Builds an HNSW index over the embeddings so similarity queries can be
    answered quickly.
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping  # noqa: UP035

import hnswlib
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.utils import timing

# Load environment variables from .env file if present
load_dotenv(override=True)


def _resolve_device(device: str) -> str:
    """Return a safe device string for the current machine.

    CUDA is only valid when PyTorch was built with CUDA support and the
    current runtime can access a GPU. Otherwise, fall back to CPU to avoid
    the AssertionError raised by SentenceTransformer/torch when CUDA is
    requested but not available.
    """
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("CUDA requested but not available on this system; "
              "using CPU instead.")
        return "cpu"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'cuda'.")


# -----------------------------------------------------------------------------
# HNSW and clustering configuration
# -----------------------------------------------------------------------------
@dataclass
class EmbeddingConfig:
    """Configuration for sentence embedding generation.

    The model is used to map each sentence into a semantic vector space.
    Normalization is enabled so cosine similarity can be computed efficiently
    with dot products.
    """

    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 256
    device: str = "cuda"
    normalize: bool = True

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device)


@dataclass
class HNSWConfig:
    """Configuration for the HNSW approximate nearest-neighbor index.

    HNSW builds a graph with layered long-range links, allowing fast
    approximate nearest neighbor queries even for large embedding sets.
    The graph stores cosine-similarity relationships between sentence vectors.
    """

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
    """Configuration for the Leiden clustering stage and recursive hierarchy.

    Leiden is a graph-based community detection algorithm that partitions
    nodes into dense, semantically related communities. The resolution
    parameter controls partition granularity, while the hierarchy settings
    decide when to keep splitting sub-clusters.
    """

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
    """
    Top-level configuration bundle for the end-to-end clustering pipeline.
    """

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    leiden: LeidenConfig = field(default_factory=LeidenConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load pipeline settings from a YAML file."""
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as config_file:
            values = yaml.safe_load(config_file) or {}

        if not isinstance(values, Mapping):
            raise ValueError(  # noqa: TRY004
                f"Pipeline config must be a YAML mapping: {config_path}"
            )

        def section(name: str) -> dict[str, Any]:
            section_values = values.get(name, {})
            if not isinstance(section_values, Mapping):
                raise ValueError(  # noqa: TRY004
                    f"Pipeline config section '{name}' must be a mapping"
                )
            return dict(section_values)

        return cls(
            embedding=EmbeddingConfig(**section("embedding")),
            hnsw=HNSWConfig(**section("hnsw")),
            leiden=LeidenConfig(**section("leiden")),
        )


# ============================================================
# Hierarchy data structure
# ============================================================

@dataclass
class ClusterNode:
    """Node in the hierarchical clustering tree.

    Each node represents a set of sentence indices that belong to the same
    semantic cluster.
    The tree is built recursively: every cluster can be subdivided into child
    clusters based on graph community structure. The node stores both the raw
    sentence membership and a set of representative sentence indices for
    summary output.
    """

    cluster_id: str
    depth: int
    indices: np.ndarray
    resolution: float

    children: list[ClusterNode] = field(default_factory=list)

    # Useful descriptive information
    representative_indices: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Return the number of sentence indices contained in this cluster."""
        return len(self.indices)

    @property
    def is_leaf(self) -> bool:
        """
        Return True when this node has no children and therefore terminates\
        a branch.
        """
        return len(self.children) == 0


# ============================================================
# Embeddings
# ============================================================

@timing
def embed_sentences(
    sentences: list[str],
    config: EmbeddingConfig,
) -> np.ndarray:
    """Encode a list of sentences into a dense embedding matrix.

    The function loads a SentenceTransformer model, runs batch inference over
    the corpus, and returns an N x D float32 array. When configured,
    embeddings are normalized so that cosine similarity can be computed
    efficiently using dot products.

    Args:
        sentences: text snippets to embed.
        config: embedding model settings, including the model name, device,
                batch size, and whether to normalize vectors.

    Returns:
        A NumPy array of shape (n_sentences, embedding_dim) containing
        normalized embeddings.
    """

    config.device = _resolve_device(config.device)

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

@timing
def build_hnsw_index(
    embeddings: np.ndarray,
    config: HNSWConfig,
) -> hnswlib.Index:
    """
    Construct a hierarchical navigable small-world graph over the
    embedding set.

    HNSW organizes vectors into a layered graph where each node has
    short-range links for local structure and long-range links for efficient
    navigation. The algorithm approximates the k-nearest neighbors much faster
    than exhaustive search while retaining high recall, making it suitable for
    large semantic corpora.

    Args:
        embeddings: matrix of sentence vectors.
        config: HNSW settings such as the number of neighbors and
                search efficiency.

    Returns:
        A configured hnswlib index ready for approximate nearest-neighbor
        queries.
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
