"""Gemini-backed knowledge-gap expansion for DecisionRAG."""
from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..schemas.recommendation import RecommendationRequest, RecommendationResponse

if TYPE_CHECKING:
    from .engine import DecisionEngine


_SYSTEM_INSTRUCTION = """\
You are a factual knowledge-tree gardener. Return only a JSON object representing
a dataset root with id, title, description, level, metadata, and children. Every
node must have id, title, description, level, metadata, and children when it has
children. Use metadata as an object.

Answer the user's query with accurate, real-world entities and concise factual
descriptions. At the lowest level, include exactly 3 to 5 real, specific works,
books, products, or people that directly answer the query. Use their actual names
as leaf titles and include useful author, creator, year, or synopsis details in
metadata and description where known. Do not invent facts; omit uncertain details.

Never return generic or fabricated placeholders, including phrases such as
"〜の領域", "〜標準仕様", "記念碑的代表作", "関連領域", "コアアイテム",
or generic titles in place of real names. Do not claim that a generated title is
an actual work unless it is real. Return valid JSON only, with no markdown fences.
"""

_FORBIDDEN_LEAF_TERMS = (
    "〜",
    "標準仕様",
    "記念碑的代表作",
    "関連領域",
    "コアアイテム",
    "詳細解説",
    "テンプレート",
)


class TreeGardener:
    """Uses Gemini to generate and validate a concrete knowledge-tree branch."""

    model_name = "gemini-2.5-flash"

    def cultivate(self, query: str) -> dict[str, Any]:
        """Generate a mountable tree branch for a knowledge-gap query."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required to grow the knowledge tree."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini tree generation requires the google-genai package."
            ) from exc

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        response = client.models.generate_content(
            model=self.model_name,
            contents=f"Create a factual knowledge tree that answers this query: {query}",
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            ),
        )
        response_text = response.text
        if not response_text:
            raise ValueError("Gemini returned an empty knowledge-tree response.")
        try:
            branch = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid JSON for the knowledge tree.") from exc
        return self._validate_and_namespace(branch, query)

    @staticmethod
    def _validate_and_namespace(branch: Any, query: str) -> dict[str, Any]:
        if not isinstance(branch, dict):
            raise TypeError("Gemini knowledge-tree response must be a JSON object.")

        def validate_node(node: Any) -> list[dict[str, Any]]:
            if not isinstance(node, dict):
                raise TypeError("Every Gemini tree node must be a JSON object.")
            for key in ("id", "title", "description", "level", "metadata"):
                if key not in node:
                    raise ValueError(f"Gemini tree node is missing required field: {key}")
            if not isinstance(node["metadata"], dict):
                raise TypeError("Gemini tree node metadata must be a JSON object.")
            if not isinstance(node["title"], str) or not node["title"].strip():
                raise ValueError("Gemini tree node titles must be non-empty strings.")
            children = node.get("children", [])
            if not isinstance(children, list):
                raise TypeError("Gemini tree node children must be a JSON array.")
            leaves: list[dict[str, Any]] = []
            if children:
                for child in children:
                    leaves.extend(validate_node(child))
            else:
                leaves.append(node)
            return leaves

        leaves = validate_node(branch)
        if not branch.get("children"):
            raise ValueError("Gemini must return a tree root with children.")
        if not 3 <= len(leaves) <= 5:
            raise ValueError(
                f"Gemini must return 3 to 5 concrete leaf items; received {len(leaves)}."
            )
        for leaf in leaves:
            title = leaf["title"].strip()
            if any(term in title for term in _FORBIDDEN_LEAF_TERMS):
                raise ValueError(
                    f"Gemini returned a placeholder instead of a real leaf item: {title}"
                )
        if len({leaf["title"].strip().casefold() for leaf in leaves}) != len(leaves):
            raise ValueError("Gemini returned duplicate leaf item titles.")

        branch = json.loads(json.dumps(branch, ensure_ascii=False))
        dataset_id = f"gemini_{uuid.uuid4().hex}"

        def namespace(node: dict[str, Any], path: str) -> None:
            node["id"] = f"{dataset_id}_{path}"
            node.setdefault("description", "")
            node.setdefault("metadata", {})
            node.setdefault("level", 1)
            for index, child in enumerate(node.get("children", [])):
                namespace(child, f"{path}_{index}")

        namespace(branch, "root")
        branch["title"] = str(branch["title"]).strip()
        branch["description"] = str(branch.get("description", "")).strip()
        branch["metadata"]["source"] = "Google Gemini"
        branch["metadata"]["query"] = query
        return branch


@dataclass
class KnowledgeGapEvent:
    """Audit record of a detected knowledge gap and resulting tree graft."""

    event_id: str
    timestamp: str
    query: str
    top_score_before: float
    top_title_before: str
    gap_detected: bool
    missing_concept: str
    grafted_dataset_id: str | None = None
    grafted_title: str | None = None
    grafted_nodes_count: int = 0
    top_score_after: float | None = None
    top_title_after: str | None = None
    accuracy_gain_pct: float = 0.0
    status: str = "completed"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "top_score_before": self.top_score_before,
            "top_title_before": self.top_title_before,
            "gap_detected": self.gap_detected,
            "missing_concept": self.missing_concept,
            "grafted_dataset_id": self.grafted_dataset_id,
            "grafted_title": self.grafted_title,
            "grafted_nodes_count": self.grafted_nodes_count,
            "top_score_after": self.top_score_after,
            "top_title_after": self.top_title_after,
            "accuracy_gain_pct": self.accuracy_gain_pct,
            "status": self.status,
            "message": self.message,
        }


class GardenerWorker:
    """Compatibility facade for callers that want an explicit graft event."""

    def __init__(self, gap_threshold: float = 0.30):
        self.gap_threshold = gap_threshold
        self.evolution_log: list[KnowledgeGapEvent] = []
        self.tree_gardener = TreeGardener()

    def evaluate_gap(
        self, response: RecommendationResponse
    ) -> tuple[bool, str, float, str]:
        if not response.results:
            return True, response.query, 0.0, "なし"
        top_item = response.results[0]
        return (
            top_item.score < self.gap_threshold,
            response.query if top_item.score < self.gap_threshold else "",
            top_item.score,
            top_item.title,
        )

    def _extract_missing_entity(self, query: str) -> str:
        terms = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}|[a-zA-Z0-9_-]{3,}", query)
        return " / ".join(terms[:2]) if terms else query[:40]

    def synthesize_branch(
        self, query: str, missing_concept: str | None = None
    ) -> dict[str, Any]:
        del missing_concept
        return self.tree_gardener.cultivate(query)

    def auto_graft_and_evolve(
        self,
        engine: DecisionEngine,
        query: str,
        custom_branch: dict[str, Any] | None = None,
    ) -> tuple[KnowledgeGapEvent, RecommendationResponse]:
        request = RecommendationRequest(query=query, top_k=3)
        baseline = engine.search(request)
        gap_detected, _, score_before, title_before = self.evaluate_gap(baseline)
        if not gap_detected:
            event = KnowledgeGapEvent(
                event_id=f"gap_evt_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.datetime.now().isoformat(),
                query=query,
                top_score_before=score_before,
                top_title_before=title_before,
                gap_detected=False,
                missing_concept="",
                status="skipped",
                message="知識の空白は検出されませんでした。",
            )
            self.evolution_log.append(event)
            return event, baseline

        missing_concept = self._extract_missing_entity(query)
        branch = custom_branch or self.synthesize_branch(query, missing_concept)
        mount_info = engine.mount_dataset(branch)
        after = engine.search(request)
        score_after = after.results[0].score if after.results else 0.0
        title_after = after.results[0].title if after.results else "該当なし"
        gain = round((score_after - score_before) * 100.0, 1)
        event = KnowledgeGapEvent(
            event_id=f"gap_evt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.datetime.now().isoformat(),
            query=query,
            top_score_before=round(score_before, 3),
            top_title_before=title_before,
            gap_detected=True,
            missing_concept=missing_concept,
            grafted_dataset_id=mount_info["id"],
            grafted_title=mount_info["title"],
            grafted_nodes_count=mount_info["node_count"],
            top_score_after=round(score_after, 3),
            top_title_after=title_after,
            accuracy_gain_pct=gain,
            message=(
                f"🚨 知識の空白（スコア: {score_before * 100:.0f}%）を検知。"
                f"庭師エージェントが【{mount_info['title']}】を接ぎ木し、"
                f"適合スコアが {score_after * 100:.0f}% へ変化しました。"
            ),
        )
        self.evolution_log.append(event)
        return event, after
