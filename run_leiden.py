from src.pipeline import PipelineConfig, run_pipeline
from src.tree import print_tree, save_tree
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    sentences = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "Cats and dogs can be friends.",
        "I love my pet cat.",
        "My dog loves to play fetch.",
        "Cats are independent animals.",
        "Dogs are loyal companions.",
        "I have a pet dog named Max.",
        "Cats purr when they are happy.",
        "Dogs bark to communicate.",
    ]

    config = PipelineConfig()

    result = run_pipeline(
        sentences,
        config,
    )

    print_tree(
        result["tree"],
        result["sentences"],
    )

    save_tree(
        result["tree"],
        result["sentences"],
        "topic_hierarchy.json"


if __name__ == "__main__":
    main()
    )
