import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_ai_dataset, load_books_dataset


def test_load_books_dataset_reads_json_and_filters_by_length(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(["short", "much longer sentence", "ok"]))

    rows = load_books_dataset(str(path), dataset_name="sample", max_len=10)

    assert rows == ["short", "ok"]


def test_load_ai_dataset_joins_directory_and_uses_loader(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "ai_dataset.json"
    dataset_path.write_text(json.dumps(["one", "two"]))

    called = {}

    def fake_loader(file_path, dataset_name, max_len=None):
        called["file_path"] = file_path
        called["dataset_name"] = dataset_name
        called["max_len"] = max_len
        return ["one", "two"]

    monkeypatch.setattr("src.utils.load_books_dataset", fake_loader)
    rows = load_ai_dataset(dataset_name="ai_dataset.json", dir=str(dataset_dir))

    assert rows == ["one", "two"]
    assert called["file_path"] == str(dataset_path)
    assert called["dataset_name"] == "ai_dataset.json"
    assert called["max_len"] is None
