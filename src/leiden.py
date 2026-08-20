"""Graph clustering with the Leiden algorithm and recursive cluster splitting.

This module turns the sentence similarity graph into a community hierarchy. Leiden is a
modularity-optimizing algorithm that groups highly connected nodes into dense communities. By
repeating this process on each subgraph, the project builds a hierarchical clustering tree that
captures semantic structure at multiple scales.

Functions
---------
leiden_partition
    Runs a single Leiden partition on one graph.
hierarchical_leiden
    Recursively builds the cluster hierarchy from the root graph.
_split_cluster
    Repeats the partition process on each cluster until stopping conditions are met.
"""

# ============================================================
# Leiden
# ============================================================

from __future__ import annotations  # noqa: I001

import numpy as np

from hnsw import LeidenConfig
import igraph as ig
import leidenalg as la


def leiden_partition(
    graph: ig.Graph,
    resolution: float,
    config: LeidenConfig,
) -> list[list[int]]:
    """Partition a graph into communities using the Leiden algorithm.

    Leiden optimizes a modularity objective while preserving local density, which makes it well
    suited to clustering sparse similarity graphs. The algorithm returns communities as lists of
    local vertex indices within the current subgraph.

    Args:
        graph: the similarity graph to partition.
        resolution: Leiden resolution parameter controlling cluster granularity.
        config: clustering configuration, including iteration count and seed.

    Returns:
        A list of communities, where each community is a list of local graph vertex indices.
    """

    if graph.vcount() == 0:
        return []

    if graph.ecount() == 0:
        return [list(range(graph.vcount()))]

    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=config.n_iterations,
        seed=config.seed,
    )

    communities = [
        list(community)
        for community in partition
    ]

    return communities


# ============================================================
# Hierarchical Leiden
# ============================================================

def hierarchical_leiden(
    graph: ig.Graph,
    config: LeidenConfig,
) -> ClusterNode:
    """Build a recursive clustering hierarchy from the full graph.

    The root cluster contains every sentence. The function then repeatedly applies Leiden
    partitioning to subgraphs, creating child clusters for each detected community. As the
    hierarchy deepens, the resolution increases so smaller, more specific semantic groups can be
    discovered without losing the broader structure.

    Args:
        graph: the full sentence similarity graph.
        config: Leiden and recursion settings.

    Returns:
        The root node of the hierarchical cluster tree.
    """

    all_indices = np.arange(
        graph.vcount(),
        dtype=np.int64,
    )

    root = ClusterNode(
        cluster_id="root",
        depth=0,
        indices=all_indices,
        resolution=config.resolution,
    )

    _split_cluster(
        graph=graph,
        node=root,
        config=config,
    )

    return root


def _split_cluster(
    graph: ig.Graph,
    node: ClusterNode,
    config: LeidenConfig,
) -> None:
    """Split one cluster node into child communities when the stopping conditions allow it.

    The function creates an induced subgraph for the cluster's members, applies Leiden
    partitioning, filters out tiny children, and then recurses into each accepted child cluster.
    This yields a hierarchical tree where each level reveals increasingly specific semantic
    groupings.

    Args:
        graph: the global graph containing all sentence relationships.
        node: the current cluster node to subdivide.
        config: recursive partitioning settings.
    """

    # --------------------------------------------------------
    # Stopping criteria
    # --------------------------------------------------------

    if node.depth >= config.max_depth:
        return

    if node.size < config.min_cluster_size:
        return

    # Build induced subgraph for this cluster.
    subgraph = graph.induced_subgraph(
        node.indices.tolist()
    )

    if subgraph.ecount() == 0:
        return

    # Increase resolution as hierarchy becomes deeper.
    child_resolution = (
        node.resolution
        if node.depth == 0
        else node.resolution
        * config.resolution_multiplier
    )

    communities = leiden_partition(
        subgraph,
        resolution=child_resolution,
        config=config,
    )

    # No meaningful subdivision.
    if len(communities) <= 1:
        return

    # --------------------------------------------------------
    # Map local subgraph indices back to global indices.
    # --------------------------------------------------------

    children = []

    for community_number, local_indices in enumerate(
        communities
    ):
        global_indices = node.indices[
            np.asarray(
                local_indices,
                dtype=np.int64,
            )
        ]

        if len(global_indices) < config.min_child_size:
            continue

        child_id = (
            f"{node.cluster_id}."
            f"{community_number}"
        )

        child = ClusterNode(
            cluster_id=child_id,
            depth=node.depth + 1,
            indices=global_indices,
            resolution=child_resolution,
        )

        children.append(child)

    # If filtering removed almost everything,
    # don't create a broken hierarchy.
    if len(children) <= 1:
        return

    node.children = children

    # --------------------------------------------------------
    # Recurse
    # --------------------------------------------------------

    for child in children:
        _split_cluster(
            graph=graph,
            node=child,
            config=config,
        )
