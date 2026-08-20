import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cluster import add_representatives, node_to_dict, print_tree, save_tree
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


def test_print_tree_captures_output(capsys):
    child = ClusterNode(cluster_id="root.child", depth=1, indices=np.array([0], dtype=np.int64), resolution=0.5)
    root = ClusterNode(cluster_id="root", depth=0, indices=np.array([0], dtype=np.int64), resolution=1.0, children=[child])
    root.representative_indices = [0]
    child.representative_indices = [0]

    print_tree(root, ["one"], max_examples=1)

    captured = capsys.readouterr().out
    assert "root" in captured
    assert "root.child" in captured
    assert "one" in captured
