from argparse import ArgumentParser

from dotenv import load_dotenv
from torch import device

from src.cluster import print_tree, save_tree
from src.hnsw import PipelineConfig
from src.pipeline import run_pipeline
from src.utils import load_ai_dataset, timing

load_dotenv(override=True)


# -------------------------------------------------------------------------------\
# Run Leiden Pipeline
# -------------------------------------------------------------------------------

def run_leiden_pipeline(sentences: list[str], config: PipelineConfig,
                        device: str = "cpu",
                        output_path: str = "topic_hierarchy.json",
                        verbose: bool = True
                        ) -> dict[str, object]:
    """
    Run the full Leiden topic-clustering pipeline on a corpus of sentences.
    """
    config.embedding.device = device
    result = run_pipeline(sentences, config)

    if verbose:
        print_tree(
            result["tree"],
            result["sentences"],
        )

    save_tree(
        result["tree"],
        result["sentences"],
        output_path,
    )

    print(f"\nSaved topic tree to: {output_path}")

    return result


# -----------------------------------------------------------------------------
# Command-Line Interface
# -----------------------------------------------------------------------------

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run the Leiden topic-clustering pipeline on a dataset.",
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default="ai_dataset.json",
        help="Name of the dataset to load from the configured data directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="topic_hierarchy.json",
        help="Optional JSON path to save the resulting topic tree.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for sentence embeddings. Default is CPU.",
    )
    return parser


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

@timing
def main() -> None:
    args = parse_args().parse_args()

    sentences = load_ai_dataset(
        dataset_name=args.dataset_name,
    )

    if not sentences:
        raise ValueError("No sentences were loaded from dataset: "
                         f"{args.dataset_name}")

    config = PipelineConfig()
    run_leiden_pipeline(
        sentences=sentences,
        config=config,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
