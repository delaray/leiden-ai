# ============================================================
# Cluster representatives
# ============================================================

import json

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


def print_tree(
    node: ClusterNode,
    sentences: list[str],
    max_examples: int = 2,
) -> None:

    indent = "    " * node.depth

    print(
        f"{indent}"
        f"{node.cluster_id} "
        f"[{node.size:,} sentences]"
    )

    for idx in node.representative_indices[
        :max_examples
    ]:
        sentence = sentences[idx]

        print(
            f"{indent}    • {sentence}"
        )

    for child in node.children:
        print_tree(
            child,
            sentences,
            max_examples,
        )

def node_to_dict(
    node: ClusterNode,
    sentences: list[str],
) -> dict:

    return {
        "cluster_id": node.cluster_id,
        "depth": node.depth,
        "size": node.size,
        "resolution": node.resolution,

        "representative_sentences": [
            sentences[i]
            for i in node.representative_indices
        ],

        "sentence_indices": (
            node.indices.tolist()
            if node.is_leaf
            else None
        ),

        "children": [
            node_to_dict(child, sentences)
            for child in node.children
        ],
    }


def save_tree(
    root: ClusterNode,
    sentences: list[str],
    filename: str | Path,
) -> None:

    data = node_to_dict(
        root,
        sentences,
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )
