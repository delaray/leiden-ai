# ============================================================
# Cluster representatives
# ============================================================

import numpy as np

from hnsw import ClusterNode


def add_representatives(
    node: ClusterNode,
    embeddings: np.ndarray,
    n_representatives: int = 5,
) -> None:
    """
    Find sentences closest to the cluster centroid.
    """

    indices = node.indices

    cluster_embeddings = embeddings[indices]

    centroid = cluster_embeddings.mean(
        axis=0,
        dtype=np.float32,
    )

    norm = np.linalg.norm(centroid)

    if norm > 0:
        centroid /= norm

    # Embeddings are normalized, therefore dot product
    # is cosine similarity.
    similarities = (
        cluster_embeddings @ centroid
    )

    n = min(
        n_representatives,
        len(indices),
    )

    best_local = np.argpartition(
        -similarities,
        kth=n - 1,
    )[:n]

    best_local = best_local[
        np.argsort(
            -similarities[best_local]
        )
    ]

    node.representative_indices = (
        indices[best_local].tolist()
    )

    for child in node.children:
        add_representatives(
            child,
            embeddings,
            n_representatives,
        )
