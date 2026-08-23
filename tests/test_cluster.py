import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cluster import (
    add_representatives,
    load_taxonomy,
    node_to_dict,
    print_tree,
    save_taxonomy,
    save_tree,
)
from src.hnsw import ClusterNode


def test_add_representatives_selects_central_indices():
    node = ClusterNode(cluster_id="root", depth=0, indices=np.array([0, 1, 2], dtype=np.int64), resolution=1.0)
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    add_representatives(node, embeddings, n_representatives=2)

    assert node.representative_indices == [1, 2]
    assert len(node.representative_indices) == 2


def test_node_to_dict_and_save_tree(tmp_path):
    child = ClusterNode(cluster_id="root.a", depth=1, indices=np.array([0, 1], dtype=np.int64), resolution=0.5)
    root = ClusterNode(cluster_id="root", depth=0, indices=np.array([0, 1], dtype=np.int64), resolution=1.0, children=[child])
    root.representative_indices = [0]
    child.representative_indices = [1]

    data = node_to_dict(root, ["alpha", "beta"])
    assert data["cluster_id"] == "root"
    assert data["children"][0]["cluster_id"] == "root.a"
    assert data["representative_sentences"] == ["alpha"]

    out_path = tmp_path / "tree.json"
    save_tree(root, ["alpha", "beta"], out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["cluster_id"] == "root"
    assert loaded["children"][0]["cluster_id"] == "root.a"


def test_print_tree_hides_representatives_by_default(capsys):
    child = ClusterNode(cluster_id="root.child", depth=1, indices=np.array([0], dtype=np.int64), resolution=0.5)
    root = ClusterNode(cluster_id="root", depth=0, indices=np.array([0], dtype=np.int64), resolution=1.0, children=[child])
    root.representative_indices = [0]
    child.representative_indices = [0]

    print_tree(root, ["one"])

    captured = capsys.readouterr().out
    assert "root" in captured
    assert "root.child" in captured
    assert "one" not in captured


def test_print_tree_can_show_representatives(capsys):
    root = ClusterNode(cluster_id="root", depth=0, indices=np.array([0], dtype=np.int64), resolution=1.0)
    root.representative_indices = [0]

    print_tree(root, ["one"], max_examples=1, show_representatives=True)

    captured = capsys.readouterr().out
    assert "one" in captured


def test_save_and_load_taxonomy(tmp_path):
    child = ClusterNode(
        cluster_id="root.child",
        depth=1,
        indices=np.array([0], dtype=np.int64),
        resolution=0.5,
        label="Child topic",
    )
    root = ClusterNode(
        cluster_id="root",
        depth=0,
        indices=np.array([0, 1], dtype=np.int64),
        resolution=1.0,
        children=[child],
        label="Root topic",
    )
    root.representative_indices = [1]
    child.representative_indices = [0]
    out_path = tmp_path / "taxonomy.json"

    save_taxonomy(root, out_path)
    taxonomy = load_taxonomy(out_path)

    assert taxonomy == {
        "name": "Root topic",
        "size": 2,
        "children": [
            {
                "name": "Child topic",
                "size": 1,
                "children": [],
            },
        ],
    }
    assert "representative_sentences" not in out_path.read_text(encoding="utf-8")


def test_save_taxonomy_uses_print_tree_unlabeled_fallback(tmp_path):
    root = ClusterNode(
        cluster_id="root",
        depth=0,
        indices=np.array([0], dtype=np.int64),
        resolution=1.0,
    )
    out_path = tmp_path / "taxonomy.json"

    save_taxonomy(root, out_path)

    assert load_taxonomy(out_path)["name"] == "Unlabeled"


def test_generate_cluster_label_disables_thinking(monkeypatch):
    from src.cluster import generate_cluster_label
    from src.hnsw import LabelingConfig

    request_json = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "Vector Search"}

    def fake_post(url, json, timeout):
        request_json.update(json)
        return FakeResponse()

    monkeypatch.setattr("src.cluster.requests.post", fake_post)

    label = generate_cluster_label(
        ["HNSW retrieves similar embedding vectors."],
        LabelingConfig(),
    )

    assert label == "Vector Search"
    assert request_json["think"] is False
