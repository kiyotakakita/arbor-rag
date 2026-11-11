"""Unit tests for Auto-Gardener self-growing knowledge worker in DecisionRAG."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.decision_model import MockDecisionModel
from src.core.dense_retriever import DenseRetriever
from src.core.engine import DecisionEngine
from src.core.gardener import GardenerWorker, KnowledgeGapEvent, TreeGardener
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

    def _ochiai_branch(self):
        return {
            "id": "test_ochiai",
            "title": "落合陽一の著作",
            "description": "落合陽一によるメディア論とデジタルネイチャーの著作",
            "level": 1,
            "metadata": {},
            "children": [
                {
                    "id": "media_theory",
                    "title": "メディア論",
                    "description": "落合陽一の著作と思想",
                    "level": 2,
                    "metadata": {},
                    "children": [
                        {
                            "id": "digital_nature",
                            "title": "デジタルネイチャー (Digital Nature)",
                            "description": "落合陽一著。計算機自然を論じた著作。",
                            "level": 3,
                            "metadata": {
                                "author": "落合陽一",
                                "keywords": ["デジタルネイチャー", "計算機自然"],
                            },
                        },
                        {
                            "id": "magic_century",
                            "title": "魔法の世紀",
                            "description": "落合陽一によるテクノロジー論。",
                            "level": 3,
                            "metadata": {"author": "落合陽一"},
                        },
                        {
                            "id": "mass_of_media",
                            "title": "質量への憧憬",
                            "description": "落合陽一によるメディアアートの著作。",
                            "level": 3,
                            "metadata": {"author": "落合陽一"},
                        },
                    ],
                }
            ],
        }

    def test_gap_detection_on_unknown_query(self):
        """Searching for Ochiai Yoichi before mounting should trigger gap detection (< 50%)."""
        res = self.engine.search(RecommendationRequest(query="落合陽一のデジタルネイチャーと計算機自然", top_k=2))
        is_gap, missing, top_score, _ = self.gardener.evaluate_gap(res)
        self.assertTrue(is_gap)
        self.assertLess(top_score, 0.50)
        self.assertIn("落合陽一", missing)

    def test_auto_graft_and_evolve(self):
        """Auto-Gardener should synthesize branch, mount it, and boost score to near 100%."""
        query = "落合陽一のデジタルネイチャーと計算機自然"
        event, after_res = self.gardener.auto_graft_and_evolve(
            self.engine, query, custom_branch=self._ochiai_branch()
        )

        self.assertIsInstance(event, KnowledgeGapEvent)
        self.assertTrue(event.gap_detected)
        self.assertIsNotNone(event.grafted_dataset_id)
        self.assertGreaterEqual(event.top_score_after, 0.90)
        self.assertIn("デジタルネイチャー", event.top_title_after)
        self.assertGreater(len(after_res.results), 0)
        self.assertEqual(after_res.results[0].title, "デジタルネイチャー (Digital Nature)")

    def test_rejects_placeholder_leaf_titles(self):
        branch = self._ochiai_branch()
        branch["children"][0]["children"][0]["title"] = "富野由悠季 記念碑的代表作"
        with self.assertRaisesRegex(ValueError, "placeholder"):
            TreeGardener._validate_and_namespace(branch, "富野由悠季")

    def test_cultivate_uses_gemini_json_api(self):
        client = Mock()
        client.models.generate_content.return_value.text = json.dumps(
            self._ochiai_branch(), ensure_ascii=False
        )
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("google.genai.Client", return_value=client) as client_factory,
        ):
            branch = TreeGardener().cultivate("落合陽一の著作")

        client_factory.assert_called_once_with(api_key="test-key")
        call = client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(call.kwargs["config"].response_mime_type, "application/json")
        self.assertIn("記念碑的代表作", call.kwargs["config"].system_instruction)
        self.assertTrue(branch["id"].startswith("gemini_"))

    def test_cultivate_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(
            RuntimeError, "GEMINI_API_KEY"
        ):
            TreeGardener().cultivate("未知の概念")


if __name__ == "__main__":
    unittest.main()
