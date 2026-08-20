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
    """
    Partition an igraph graph using Leiden.

    Returns lists of LOCAL vertex indices.
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
    """
    Build a recursive Leiden hierarchy over the graph.
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
    """
    Recursively partition one ClusterNode.
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
