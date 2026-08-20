# ============================================================
# k-NN graph construction
# ============================================================

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


def build_knn_graph(
    embeddings: np.ndarray,
    index: hnswlib.Index,
    config: HNSWConfig,
    query_batch_size: int = 10_000,
) -> ig.Graph:
    """
    Build a weighted undirected k-NN graph.

    Nodes:
        sentences

    Edges:
        approximate nearest-neighbor relationships

    Weight:
        cosine similarity
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
