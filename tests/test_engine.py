"""Integration and unit tests for DecisionEngine in DecisionRAG."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.core.decision_model import MockDecisionModel
from src.core.dense_retriever import DenseRetriever
from src.core.engine import DecisionEngine
from src.core.tree import TreeIndex
from src.schemas.recommendation import RecommendationRequest


class TestEngine(unittest.TestCase):
    def setUp(self):
        catalog_path = Path(__file__).resolve().parent.parent / "data" / "sample_catalog.json"
        self.tree_index = TreeIndex.from_file(catalog_path)
        self.decision_model = MockDecisionModel()
        self.dense_retriever = DenseRetriever()
        self.engine = DecisionEngine(
            tree_index=self.tree_index,
            decision_model=self.decision_model,
            dense_retriever=self.dense_retriever,
        )

    def test_nolan_2000_query_selects_memento(self):
        """Tests that 'ノーラン監督の2000年の映画' correctly identifies and ranks Memento (2000) as #1."""
        req = RecommendationRequest(
            query="ノーラン監督の2000年の映画",
            top_k=3,
            branch_top_k=2,
            prune_threshold=0.2,
            enable_dense_hybrid=True,
            dense_top_n=2,
        )
        response = self.engine.search(req)

        self.assertGreater(len(response.results), 0)
        top_result = response.results[0]
        # Should be Memento
        self.assertEqual(top_result.id, "item_memento")
        self.assertIn("メメント", top_result.title)
        self.assertEqual(top_result.metadata.get("year"), 2000)
        self.assertIn("Nolan", str(top_result.metadata.get("director")))

        # Check trace contains Level 1, Level 2, etc.
        stages = [t.stage for t in response.trace]
        self.assertTrue(any("Level 1" in s for s in stages))
        self.assertTrue(any("Dense" in s for s in stages))

    def test_pruning_removes_unrelated_categories(self):
        """Tests that anime/fantasy is pruned when searching for hard crime thrillers."""
        req = RecommendationRequest(
            query="七つの大罪を模したサイコスリラー犯罪事件 デヴィッド・フィンチャー",
            top_k=3,
            branch_top_k=1,
            prune_threshold=0.2,
            enable_dense_hybrid=False,
            dense_top_n=0,
        )
        response = self.engine.search(req)

        self.assertGreater(len(response.results), 0)
        top_titles = [r.title for r in response.results]
        self.assertTrue(any("セブン" in t for t in top_titles))

        # Check that anime category was pruned in trace
        l1_step = response.trace[0]
        anime_node_eval = next(
            (e for e in l1_step.evaluated if "anime" in e.id or "アニメ" in e.title), None
        )
        if anime_node_eval:
            self.assertFalse(anime_node_eval.survived)

    def test_dense_hybrid_rescue(self):
        """Tests that dense retrieval rescues relevant items even when tree navigation branches differently."""
        req = RecommendationRequest(
            query="夢探偵パプリカ 平沢進 DCミニ",
            top_k=3,
            branch_top_k=2,
            prune_threshold=0.1,
            enable_dense_hybrid=True,
            dense_top_n=3,
        )
        response = self.engine.search(req)
        result_ids = [r.id for r in response.results]
        self.assertIn("item_paprika", result_ids)

    def test_dynamic_dataset_mount_and_search(self):
        """Tests that mounting an Ochiai Yoichi book dataset allows '落合陽一' query to rank Digital Nature #1."""
        ochiai_dataset = {
            "id": "dataset_books_ochiai",
            "title": "落合陽一書籍コレクション",
            "description": "落合陽一によるメディア論・現代思想",
            "children": [
                {
                    "id": "sub_media_thought",
                    "title": "メディア論・計算機自然",
                    "description": "デジタルネイチャーと計算機自然論",
                    "children": [
                        {
                            "id": "book_digital_nature",
                            "title": "デジタルネイチャー (Digital Nature)",
                            "description": "落合陽一著。計算機によって再構築される新しい自然観。",
                            "metadata": {
                                "author": "落合陽一",
                                "year": 2018,
                                "tags": ["思想", "AI", "計算機自然"],
                            },
                        }
                    ],
                }
            ],
        }

        # 1. Mount dataset dynamically
        self.engine.mount_dataset(ochiai_dataset)

        # 2. Search for Ochiai
        req = RecommendationRequest(
            query="落合陽一のデジタルネイチャーと計算機自然",
            top_k=3,
            branch_top_k=2,
            prune_threshold=0.2,
        )
        response = self.engine.search(req)
        self.assertGreater(len(response.results), 0)
        top_result = response.results[0]
        self.assertEqual(top_result.id, "book_digital_nature")
        self.assertEqual(top_result.metadata.get("author"), "落合陽一")

        # 3. Unmount dataset
        self.engine.unmount_dataset("dataset_books_ochiai")
        # Searching again should no longer find Digital Nature
        response_after = self.engine.search(req)
        ids_after = [r.id for r in response_after.results]
        self.assertNotIn("book_digital_nature", ids_after)


if __name__ == "__main__":
    unittest.main()
