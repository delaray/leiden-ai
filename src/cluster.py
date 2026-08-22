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
import requests

from src.hnsw import ClusterNode, LabelingConfig


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



# ============================================================
# Cluster labeling with Ollama
# ============================================================

# -----------------------------------------------------------------------------
# Generate Cluster Label
# -----------------------------------------------------------------------------

def generate_cluster_label(
    representative_sentences: list[str],
    config: LabelingConfig,
    parent_label: str | None = None,
) -> str:
    """
    Ask an Ollama model to produce a concise semantic label
    describing a cluster.
    """

    examples = "\n".join(
        f"- {sentence}"
        for sentence in representative_sentences[:config.n_examples]
    )

    parent_context = ""

    if parent_label:
        parent_context = (
            f"\nThe parent topic is: {parent_label}\n"
            "The new label should describe the more specific "
            "subtopic represented by these sentences.\n"
        )

    prompt = f"""
        You are labeling semantic clusters extracted from a technical corpus.

        Your task is to identify the single concept or topic that best
        describes the following sentences.

        {parent_context}

        Representative sentences:

        {examples}

        Requirements:

        - Return ONLY the label.
        - Do not explain your answer.
        - Use at most {config.max_label_words} words.
        - Prefer a precise technical concept over a generic description.
        - Do not use phrases such as "Discussion of", "Topics about",
        "Information on", or "Sentences about".
        - Use standard terminology when possible.

        Examples of good labels:

        Retrieval-Augmented Generation
        Grouped Query Attention
        Policy Gradient Methods
        Vector Similarity Search
        Transformer Positional Encoding
        Knowledge Graph Construction
        Contrastive Representation Learning

        Label:
        """.strip()

    response = requests.post(
        f"{config.ollama_url}/api/generate",
        json={
            "model": config.model,
            "prompt": prompt,
            "stream": False,

            # Low temperature is desirable because labeling
            # should be deterministic rather than creative.
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=config.timeout,
    )

    response.raise_for_status()

    label = response.json()["response"].strip()

    # Defensive cleanup in case the model adds quotes.
    label = label.strip('"').strip("'").strip()

    return label


# ------------------------------------------------------------------------------
# Label Cluster Tree
# ------------------------------------------------------------------------------

def label_cluster_tree(
    node: ClusterNode,
    sentences: list[str],
    config: LabelingConfig,
    parent_label: str | None = None,
) -> None:
    """
    Recursively generate human-readable labels for every cluster.
    """

    if (
        config.max_depth is not None
        and node.depth > config.max_depth
    ):
        return

    if node.size < config.min_cluster_size:
        return

    representative_sentences = [
        sentences[i]
        for i in node.representative_indices[
            :config.n_examples
        ]
    ]

    if representative_sentences:

        try:
            node.label = generate_cluster_label(
                representative_sentences,
                config=config,
                parent_label=parent_label,
            )

        except Exception as exc:  # noqa: BLE001

            print(
                f"Could not label {node.cluster_id}: {exc}"
            )

            node.label = None

    for child in node.children:

        label_cluster_tree(
            child,
            sentences=sentences,
            config=config,
            parent_label=node.label,
        )


# -----------------------------------------------------------------------------
# Print Cluster Tree
# -----------------------------------------------------------------------------

def print_tree(
    node: ClusterNode,
    sentences: list[str],
    max_examples: int = 2,
) -> None:

    indent = "    " * node.depth

    label = node.label or "Unlabeled"

    print(
        f"{indent}"
        f"{label} "
        f"[{node.cluster_id}, "
        f"{node.size:,} sentences]"
    )

    for idx in node.representative_indices[
        :max_examples
    ]:

        print(
            f"{indent}    • {sentences[idx]}"
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

    return {
        "cluster_id": node.cluster_id,

        "label": node.label,

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
            node_to_dict(
                child,
                sentences,
            )
            for child in node.children
        ],
    }


# -----------------------------------------------------------------------------
# Save Tree
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# End of File
# -----------------------------------------------------------------------------
