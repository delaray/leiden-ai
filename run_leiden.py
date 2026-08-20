from argparse import ArgumentParser

from dotenv import load_dotenv

from src.cluster import print_tree, save_tree
from src.hnsw import PipelineConfig
from src.pipeline import run_pipeline
from src.utils import load_ai_dataset

load_dotenv(override=True)


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

def main() -> None:
    args = parse_args().parse_args()

    sentences = load_ai_dataset(
        dataset_name=args.dataset_name,
    )

    if not sentences:
        raise ValueError("No sentences were loaded from dataset: "
                         f"{args.dataset_name}")

    config = PipelineConfig()
    config.embedding.device = args.device
    result = run_pipeline(sentences, config)

    print_tree(
        result["tree"],
        result["sentences"],
    )

    save_tree(
        result["tree"],
        result["sentences"],
        args.output,
    )

    print(f"Saved topic tree to: {args.output}")


if __name__ == "__main__":
    main()
