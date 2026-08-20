"""
Build a weighted k-nearest-neighbor graph from sentence embeddings.

The HNSW index provides approximate nearest neighbors efficiently. This module
converts those neighbor relations into a single undirected graph whose edges
carry cosine-similarity weights. That graph is the input to the Leiden
community detection algorithm used for hierarchical clustering.

The algorithm works by querying the ANN index in batches, keeping only edges
with sufficient similarity, canonicalizing each pair of nodes to avoid
duplicates, and retaining the strongest observed similarity for each
undirected edge.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import annotations

import hnswlib
import igraph as ig
import numpy as np
from tqdm.auto import tqdm

from src.hnsw import HNSWConfig
from src.utils import timing

# -----------------------------------------------------------------------------
# k-NN graph construction
# -----------------------------------------------------------------------------


@timing
def build_knn_graph(
    embeddings: np.ndarray,
    index: hnswlib.Index,
    config: HNSWConfig,
    query_batch_size: int = 10_000,
     ) -> ig.Graph:
    """Construct a weighted undirected k-nearest-neighbor graph.

    Each sentence is a node in the graph. For each query batch, the HNSW index
    returns the approximate nearest neighbors for every vector. The function
    converts the directional neighbors to an undirected graph by sorting each
    pair and storing the maximum similarity observed for that edge. This
    creates a similarity graph that can be partitioned by graph-community
    methods.

    Args:
        embeddings: embedding matrix for the corpus.
        index: HNSW index over those embeddings.
        config: HNSW configuration containing k and similarity threshold
                settings.
        query_batch_size: number of vectors to query in each batch.

    Returns:
        An igraph graph whose vertices are sentences and whose edge weights
        are cosine similarity.
    """

    n = len(embeddings)

    edge_weights: dict[tuple[int, int], float] = {}

    query_k = config.k + 1

    print(
        f"Constructing {config.k}-NN graph "
        f"for {n:,} vectors..."
    )

    for start in tqdm(
        range(0, n, query_batch_size),
        desc="k-NN queries",
    ):
        end = min(start + query_batch_size, n)

        labels, distances = index.knn_query(
            embeddings[start:end],
            k=query_k,
            num_threads=config.num_threads,
        )

        for local_i, (neighbors, dists) in enumerate(
            zip(labels, distances)
        ):
            i = start + local_i

            for j, distance in zip(neighbors, dists):

                j = int(j)

                # Each vector normally returns itself.
                if i == j:
                    continue

                # For hnswlib cosine space:
                #
                #     distance = 1 - cosine_similarity
                #
                similarity = 1.0 - float(distance)

                if similarity < config.min_similarity:
                    continue

                # Make edge canonical:
                #
                # (7, 2) -> (2, 7)
                #
                u, v = sorted((i, j))

                # k-NN relations are directional.
                # We convert them into an undirected graph.
                # If encountered multiple times, retain maximum
                # observed similarity.
                old_weight = edge_weights.get((u, v))

                if (
                    old_weight is None
                    or similarity > old_weight
                ):
                    edge_weights[(u, v)] = similarity

    edges = list(edge_weights.keys())
    weights = list(edge_weights.values())

    print(f"Graph edges: {len(edges):,}")

    graph = ig.Graph(
        n=n,
        edges=edges,
        directed=False,
    )

    graph.es["weight"] = weights

    print(
        f"Graph: {graph.vcount():,} vertices, "
        f"{graph.ecount():,} edges"
    )

    return graph

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
