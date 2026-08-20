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
    """
    Load a list of sentences from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the dataset.
        dataset_name (str | None): Optional name for the dataset.

    Returns:
        list[str]: List of sentences loaded from the JSON file.
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
    """
    Load or parse AI dataset from a directory.

    Args:
        dir (str): Directory containing AI dataset
    """
    dir = os.path.join(dir, dataset_name)
    dataset = load_books_dataset(dir, dataset_name=dataset_name)

    return dataset

# ---------------------------------------------------------------------------
# End of File
# ---------------------------------------------------------------------------
