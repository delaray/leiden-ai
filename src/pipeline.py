# *****************************************************************************
# leiden-ai: Hierarchical clustering of sentences using HNSW and Leiden
# *****************************************************************************

from dotenv import load_dotenv

from cluster import add_representatives
from hnsw import PipelineConfig, build_hnsw_index, embed_sentences
from knn_graph import build_knn_graph
from leiden import hierarchical_leiden

load_dotenv(override=True)


# ------------------------------------------------------------------------------
# Run Pipeline
# ------------------------------------------------------------------------------

def run_pipeline(
    sentences: list[str],
    config: PipelineConfig | None = None
     ):
    """
    Complete semantic hierarchical clustering pipeline.
    """

    if config is None:
        config = PipelineConfig()

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
    print("Hello from leiden-ai!")


if __name__ == "__main__":
    main()
