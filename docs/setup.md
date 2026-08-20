# Setup and dependencies

This page describes the supported setup and the role of each major library in the
Leiden AI pipeline.

## Requirements

- Python 3.13 or newer
- `uv` for dependency resolution and environment management
- Enough memory to hold the embedding matrix and sparse graph
- Optional CUDA-capable GPU for faster embedding inference

## Installation

```bash
git clone <repository-url>
cd leiden-ai
uv sync
```

`uv sync` creates or updates `.venv` from `pyproject.toml` and the committed
`uv.lock`. Activate it manually with `source .venv/bin/activate`, or prefix commands
with `uv run`.

## Major dependencies

| Library | Pipeline role | Why it is used |
|---|---|---|
| `sentence-transformers` | Text embedding | Provides pretrained transformer models and batched sentence encoding |
| `numpy` | Numerical arrays | Stores embeddings and performs centroid, normalization, and similarity operations |
| `hnswlib` | Approximate neighbor search | Builds a fast cosine-space HNSW index for large vector collections |
| `igraph` | Graph representation | Stores the sparse, weighted sentence-similarity graph and induced subgraphs |
| `leidenalg` | Community detection | Optimizes weighted graph partitions used to construct the hierarchy |
| `python-dotenv` | Configuration | Loads local environment variables such as `DATA_DIR` from `.env` |
| `pre-commit` | Development quality | Runs configured checks before commits |

`tqdm` is imported by the graph builder for query progress reporting. It is installed
transitively by `sentence-transformers`; declare it directly in `pyproject.toml` if the
graph module is ever packaged independently.

## CPU configuration

The library defaults to CUDA for embedding generation. Set the device explicitly when
running on a CPU-only machine:

```python
from src.hnsw import PipelineConfig

config = PipelineConfig()
config.embedding.device = "cpu"
```

The included notebook uses `sentence-transformers/all-MiniLM-L6-v2`, which is smaller
and faster than the project default while retaining good semantic behavior for a demo.
The first use of any model may download weights from Hugging Face.

## Dataset location

The CLI expects a JSON array of strings. Configure its base directory in `.env`:

```dotenv
DATA_DIR=/absolute/path/to/data
```

The resolved input directory is `$DATA_DIR/books/datasets`. For programmatic use,
passing a sentence list directly to `run_pipeline` avoids this directory convention.

## Verification

Run the unit tests and code-quality hooks after installation:

```bash
uv run --with pytest pytest
uv run pre-commit run --all-files
```

To execute the tutorial without Jupyter, install a notebook runner such as JupyterLab
or use an IDE that supports `.ipynb` files. The notebook itself requires only the
project runtime dependencies.

## Resource considerations

| Resource | Primary driver | Mitigation |
|---|---|---|
| Model memory | Embedding checkpoint and batch size | Use a smaller model or reduce `batch_size` |
| Embedding memory | Sentence count × embedding dimension | Process corpora in batches and persist vectors |
| HNSW memory | Corpus size and `M` | Reduce `M` for a smaller index |
| Graph memory | Retained edges (`k` and threshold) | Reduce `k` or raise `min_similarity` |
| Runtime | Model inference, HNSW parameters, Leiden recursion | Use GPU embeddings or tune search/depth settings |

[Back to README](../README.md)
