"""Unit tests for Auto-Gardener self-growing knowledge worker in DecisionRAG."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.core.decision_model import MockDecisionModel
from src.core.dense_retriever import DenseRetriever
from src.core.engine import DecisionEngine
from src.core.gardener import GardenerWorker, KnowledgeGapEvent
from src.core.tree import TreeIndex
from src.schemas.recommendation import RecommendationRequest


class TestGardener(unittest.TestCase):
    def setUp(self):
        catalog_path = Path(__file__).resolve().parent.parent / "data" / "sample_catalog.json"
        self.tree_index = TreeIndex.from_file(catalog_path)
        self.engine = DecisionEngine(
            tree_index=self.tree_index,
            decision_model=MockDecisionModel(),
            dense_retriever=DenseRetriever(),
        )
        self.gardener = GardenerWorker(gap_threshold=0.50)

    def test_gap_detection_on_unknown_query(self):
        """Searching for Ochiai Yoichi before mounting should trigger gap detection (< 50%)."""
        res = self.engine.search(RecommendationRequest(query="落合陽一のデジタルネイチャーと計算機自然", top_k=2))
        is_gap, missing, top_score, top_title = self.gardener.evaluate_gap(res)
        self.assertTrue(is_gap)
        self.assertLess(top_score, 0.50)
        self.assertIn("落合陽一", missing)

    def test_auto_graft_and_evolve(self):
        """Auto-Gardener should synthesize branch, mount it, and boost score to near 100%."""
        query = "落合陽一のデジタルネイチャーと計算機自然"
        event, after_res = self.gardener.auto_graft_and_evolve(self.engine, query)

        self.assertIsInstance(event, KnowledgeGapEvent)
        self.assertTrue(event.gap_detected)
        self.assertIsNotNone(event.grafted_dataset_id)
        self.assertGreaterEqual(event.top_score_after, 0.90)
        self.assertIn("デジタルネイチャー", event.top_title_after)
        self.assertGreater(len(after_res.results), 0)
        self.assertEqual(after_res.results[0].title, "デジタルネイチャー (Digital Nature)")


if __name__ == "__main__":
    unittest.main()
