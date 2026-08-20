"""Utilities for locating and loading sentence datasets used by the clustering pipeline.

This module centralizes the data-directory configuration and provides small helpers for reading
JSON-encoded sentence corpora. It is intentionally lightweight: it exposes dataset paths and
standard file-loading functions so the rest of the pipeline can focus on modeling and clustering.

Functions
---------
load_books_dataset
    Read a JSON file containing sentences and optionally filter by maximum length.
load_ai_dataset
    Load a named AI dataset from the configured data directory.
"""

import json
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Default Directories
# ---------------------------------------------------------------------------

DATA_DIR = os.getenv('DATA_DIR', '/home/pierre/projects/data')
BOOKS_DATA_DIR = os.path.join(DATA_DIR, 'books')
BOOKS_DATASETS_DIR = os.path.join(BOOKS_DATA_DIR, 'datasets')

# Load environment variables from .env file if present
load_dotenv(override=True)


# -----------------------------------------------------------------------------
# Load Books Dataset
# -----------------------------------------------------------------------------

def load_books_dataset(file_path: str, dataset_name: str,
                       max_len: int | None = None
                       ) -> list[str]:
    """Load a JSON file of sentences into memory.

    The dataset is expected to be a list of strings, one sentence per item. The optional
    max_len filter removes sentences longer than the given character count, which is useful for
    keeping corpora consistent before embedding.

    Args:
        file_path: path to the JSON file containing the sentence list.
        dataset_name: human-readable dataset name used in error messages.
        max_len: optional maximum sentence length to retain.

    Returns:
        A list of sentences loaded from the file, or an empty list if loading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        if max_len is not None:
            sentences = [s for s in sentences if len(s) <= max_len]
        print(f"Loaded {len(sentences)} sentences from:\n{file_path}")
        return sentences

    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading dataset {dataset_name} from {file_path}\n{e}")
        return []



# ---------------------------------------------------------------------------
# Load AI Dataset
# ---------------------------------------------------------------------------

def load_ai_dataset(dataset_name: str = 'ai_dataset.json',
                    dir: str = BOOKS_DATASETS_DIR
                    ) -> list[str]:
    """Load a named AI dataset from the configured datasets directory.

    This function composes the file path from the dataset directory and then delegates to
    load_books_dataset for the actual JSON parsing. It provides a simple, consistent data-loading
    interface for pipeline inputs.

    Args:
        dataset_name: file name of the dataset to load.
        dir: directory path containing the dataset.

    Returns:
        The sentence list loaded from the dataset file.
    """
    dir = os.path.join(dir, dataset_name)
    dataset = load_books_dataset(dir, dataset_name=dataset_name)

    return dataset

# ---------------------------------------------------------------------------
# End of File
# ---------------------------------------------------------------------------
