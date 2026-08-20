"""Utilities for summarizing and exporting hierarchical cluster trees.

This module is responsible for selecting representative sentences for each
semantic cluster and serializing the hierarchy for downstream inspection. The
underlying algorithm computes each cluster centroid in embedding space,
measures similarity of all member vectors to that centroid, then selects the
most central sentences as representatives.

Classes
-------
ClusterNode
    Data structure used to represent a hierarchical cluster node.

Functions
---------
add_representatives
    Finds the most central sentences within each cluster and stores
    their indices.
print_tree
    Prints a human-readable summary of the cluster hierarchy.
node_to_dict
    Converts a cluster tree to a nested dictionary for JSON export.
save_tree
    Saves the cluster hierarchy to disk as JSON.
"""

# ============================================================
# Cluster representatives
# ============================================================

import json
from pathlib import Path

import numpy as np

from hnsw import ClusterNode


def add_representatives(
    node: ClusterNode,
    embeddings: np.ndarray,
    n_representatives: int = 5,
    ) -> None:
    """Populate a cluster node with its most representative sentence indices.

    The function computes the centroid of all embedding vectors within the
    cluster, normalizes it, and measures cosine similarity between each member
    vector and the centroid. The strongest matches are chosen as
    representative sentences, which summarize the cluster meaning.

    Args:
        node: cluster node to annotate.
        embeddings: full embedding matrix for the corpus.
        n_representatives: number of central sentence indices to save
        per cluster.
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
    """Print the cluster hierarchy as a nested summary.

    Each node is printed with its cluster id and sentence count, followed by
    representative example sentences from that cluster. The recursion walks
    the tree depth-first so the user can inspect the semantic grouping from
    broad topics to finer subtopics.

    Args:
        node: current cluster node to print.
        sentences: original sentence list used to recover representative text.
        max_examples: maximum number of representative sentences to show
        per cluster.
    """

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


# -----------------------------------------------------------------------------
# Node_To_Dict
# -----------------------------------------------------------------------------

def node_to_dict(
    node: ClusterNode,
    sentences: list[str],
) -> dict:
    """Convert a cluster tree into a nested dictionary suitable for
    JSON export.

    The returned structure includes the cluster identifier, depth, size,
    resolution, representative sentence text, and child nodes. Leaves keep
    their sentence indices while internal nodes store nested child
    dictionaries for a full hierarchical representation.

    Args:
        node: cluster node to serialize.
        sentences: original sentence list used to resolve representative text.

    Returns:
        A nested dictionary representing the cluster hierarchy.
    """

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
    """Serialize the cluster tree to a JSON file.

    The function converts the cluster hierarchy to a nested dictionary and
    writes it to disk using UTF-8 text encoding and pretty indentation. This
    makes it easier to inspect the generated semantic hierarchy outside of
    the Python runtime.

    Args:
        root: root node of the tree to export.
        sentences: original corpus used to recover the representative text.
        filename: destination path for the exported JSON file.
    """

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
