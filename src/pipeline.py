"""
End-to-end sentence clustering pipeline built from embeddings, HNSW, and
Leiden.

The workflow in this module is:
(1) Normalize and clean the input sentences,
(2) Embed them with SentenceTransformer
(3) Build an HNSW index for approximate nearest-neighbor search
(4) Convert neighbor relations into a weighted similarity graph
(5) Partition that graph hierarchically with Leiden,
(6) Select representative sentences to summarize each cluster.

This file is the orchestration layer: it wires the lower-level algorithms
together in a single, repeatable pipeline.
"""

# *****************************************************************************
# leiden-ai: Hierarchical clustering of sentences using HNSW and Leiden
# *****************************************************************************

from pathlib import Path

from dotenv import load_dotenv

from src.cluster import add_representatives, label_cluster_tree
from src.hnsw import PipelineConfig, build_hnsw_index, embed_sentences
from src.knn_graph import build_knn_graph
from src.leiden import hierarchical_leiden

load_dotenv(override=True)

DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


# ------------------------------------------------------------------------------
# Run Pipeline
# ------------------------------------------------------------------------------

def run_pipeline(
    sentences: list[str],
    config: PipelineConfig | None = None
     ):
    """Run the full semantic clustering pipeline for a corpus of sentences.

    The function validates input, embeds each sentence, builds an HNSW search
    index, creates a weighted k-NN graph from approximate neighbors, runs
    hierarchical Leiden clustering, and then selects representative sentences
    for each cluster node. The result is a dictionary containing the original
    sentences, embeddings, graph, tree, and index.

    Args:
        sentences: raw sentence list to cluster.
        config: optional pipeline configuration; defaults to a standard
                PipelineConfig instance.

    Returns:
        A dictionary containing the processed corpus, embedding matrix,
        HNSW index, similarity graph, and hierarchical cluster tree.
    """

    if config is None:
        config = PipelineConfig.from_yaml(DEFAULT_CONFIG_FILE)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence and sentence.strip()
    ]

    print(f"Corpus size: {len(sentences):,}")

    # --------------------------------------------------------
    # 1. Embeddings
    # --------------------------------------------------------

    embeddings = embed_sentences(
        sentences,
        config.embedding,
    )

    # --------------------------------------------------------
    # 2. ANN index
    # --------------------------------------------------------

    hnsw_index = build_hnsw_index(
        embeddings,
        config.hnsw,
    )

    # --------------------------------------------------------
    # 3. k-NN graph
    # --------------------------------------------------------

    graph = build_knn_graph(
        embeddings,
        hnsw_index,
        config.hnsw,
    )

    # --------------------------------------------------------
    # 4. Hierarchical Leiden
    # --------------------------------------------------------

    tree = hierarchical_leiden(
        graph,
        config.leiden,
    )

    # --------------------------------------------------------
    # 5. Representative sentences
    # --------------------------------------------------------

    add_representatives(
        tree,
        embeddings,
        n_representatives=5,
    )

    # Labels depend on the representative sentences populated above.
    label_cluster_tree(tree, sentences, config.labeling)

    return {
        "sentences": sentences,
        "embeddings": embeddings,
        "hnsw_index": hnsw_index,
        "graph": graph,
        "tree": tree,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Entry point for the pipeline module.

    This placeholder currently prints a simple startup banner. In a fuller
    application, it could parse CLI arguments and run the clustering pipeline
    over a dataset file.
    """
    print("Hello from leiden-ai!")


if __name__ == "__main__":
    main()
