import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from src.cluster import print_tree, save_taxonomy, save_tree
from src.hnsw import PipelineConfig
from src.pipeline import DEFAULT_CONFIG_FILE, run_pipeline
from src.utils import load_ai_dataset, timing

load_dotenv(override=True)

DATA_DIR = os.environ.get("DATA_DIR", "data")
TAXONOMIES_DIR = os.path.join(DATA_DIR, "taxonomies")


# -------------------------------------------------------------------------------\
# Run Leiden Pipeline
# -------------------------------------------------------------------------------

def run_leiden_pipeline(sentences: list[str], config: PipelineConfig,
                        device: str | None = None,
                        output_path: str = "topic_hierarchy.json",
                        verbose: bool = True,
                        name: str = "topic",
                        ) -> dict[str, object]:
    """
    Run the full Leiden topic-clustering pipeline on a corpus of sentences.
    """
    if device is not None:
        config.embedding.device = device

    result = run_pipeline(sentences, config)

    if verbose:
        print_tree(
            result["tree"],
            result["sentences"],
        )

    # Save the topic tree to a JSON file
    save_tree(
        result["tree"],
        result["sentences"],
        output_path,
    )
    print(f"\nSaved topic tree to: {output_path}")

    root_node = result["tree"]
    # Ensure taxonomies dir exists
    taxonomies_path = Path(TAXONOMIES_DIR)
    taxonomies_path.mkdir(parents=True, exist_ok=True)
    # Create the taxonomy file path
    taxonomy_path = \
        os.path.join(TAXONOMIES_DIR, f"{name}_taxonomy.json")
    # Save the taxonomy to a JSON file
    save_taxonomy(root_node, taxonomy_path)
    print(f"\nSaved taxonomy to: {taxonomy_path}")

    return result


# -----------------------------------------------------------------------------
# Command-Line Interface
# -----------------------------------------------------------------------------

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run the Leiden topic-clustering pipeline on a dataset.",
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE),
        help="YAML pipeline configuration file. Defaults to configs/config.yaml.",
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
        default=None,
        help="Override the embedding device from the configuration file.",
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

    config = PipelineConfig.from_yaml(args.config_file)
    run_leiden_pipeline(
        sentences=sentences,
        config=config,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
