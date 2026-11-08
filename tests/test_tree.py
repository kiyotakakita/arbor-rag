"""Unit tests for TreeIndex and TreeNode in DecisionRAG."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.core.tree import TreeIndex, TreeNode


class TestTree(unittest.TestCase):
    def setUp(self):
        catalog_path = Path(__file__).resolve().parent.parent / "data" / "sample_catalog.json"
        self.tree_index = TreeIndex.from_file(catalog_path)

    def test_tree_loaded(self):
        self.assertIsNotNone(self.tree_index.root)
        self.assertEqual(self.tree_index.root.id, "root_catalog")
        self.assertTrue(len(self.tree_index.root.children) >= 3)

    def test_level_map(self):
        level_1_nodes = self.tree_index.get_nodes_at_level(1)
        self.assertEqual(len(level_1_nodes), 3)
        titles = [n.title for n in level_1_nodes]
        self.assertTrue(any("SF" in t for t in titles))
        self.assertTrue(any("サスペンス" in t for t in titles))
        self.assertTrue(any("アニメーション" in t for t in titles))

    def test_leaves_retrieval(self):
        leaves = self.tree_index.get_all_leaves()
        self.assertGreater(len(leaves), 5)
        # All leaves should have is_leaf == True
        for leaf in leaves:
            self.assertTrue(leaf.is_leaf)

    def test_path_breadcrumbs(self):
        memento_node = self.tree_index.get_node("item_memento")
        self.assertIsNotNone(memento_node)
        path = self.tree_index.get_path_to_node("item_memento")
        path_ids = [n.id for n in path]
        self.assertIn("root_catalog", path_ids)
        self.assertIn("cat_suspense", path_ids)
        self.assertIn("sub_nolan_suspense", path_ids)
        self.assertIn("item_memento", path_ids)

    def test_context_text_generation(self):
        memento_node = self.tree_index.get_node("item_memento")
        context = memento_node.get_context_text()
        self.assertIn("メメント", context)
        self.assertIn("2000", context)
        self.assertIn("Christopher Nolan", context)


    def test_mount_and_unmount_dataset(self):
        sample_dataset = {
            "id": "test_books",
            "title": "テスト書籍カタログ",
            "description": "テスト用の書籍データセット",
            "children": [
                {
                    "id": "book_test_1",
                    "title": "デジタルネイチャー",
                    "description": "落合陽一著。計算機自然。",
                    "metadata": {"author": "落合陽一", "year": 2018},
                }
            ],
        }
        initial_leaves = len(self.tree_index.get_all_leaves())
        initial_l1 = len(self.tree_index.root.children)

        # Mount
        info = self.tree_index.mount_dataset(sample_dataset)
        self.assertEqual(info["id"], "test_books")
        self.assertEqual(len(self.tree_index.root.children), initial_l1 + 1)
        self.assertEqual(len(self.tree_index.get_all_leaves()), initial_leaves + 1)
        self.assertIsNotNone(self.tree_index.get_node("book_test_1"))

        # Unmount
        unmounted = self.tree_index.unmount_dataset("test_books")
        self.assertTrue(unmounted)
        self.assertEqual(len(self.tree_index.root.children), initial_l1)
        self.assertEqual(len(self.tree_index.get_all_leaves()), initial_leaves)
        self.assertIsNone(self.tree_index.get_node("book_test_1"))

    def test_toggle_dataset(self):
        sample_dataset = {
            "id": "toggle_test",
            "title": "トグルテスト",
            "description": "有効無効テスト用",
            "children": [
                {
                    "id": "item_toggle_1",
                    "title": "アイテム1",
                    "description": "アイテム1の説明",
                }
            ],
        }
        self.tree_index.mount_dataset(sample_dataset)
        self.assertTrue(self.tree_index.is_node_enabled("item_toggle_1"))

        # Disable
        self.tree_index.toggle_dataset("toggle_test", False)
        self.assertFalse(self.tree_index.is_node_enabled("item_toggle_1"))

        # Re-enable
        self.tree_index.toggle_dataset("toggle_test", True)
        self.assertTrue(self.tree_index.is_node_enabled("item_toggle_1"))


if __name__ == "__main__":
    unittest.main()
