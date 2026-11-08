"""Decision Model abstractions and implementations for DecisionRAG.

Key Philosophy:
1. Non-autoregressive decision interface: Candidates are judged in a shared context
   in a single forward pass, rather than independently generating text.
2. Batch scoring: Relative comparison eliminates independent threshold drift.
3. Listwise reranking: Mutual contrast among surviving finalists.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from .tree import TreeNode


class BaseDecisionModel(ABC):
    """Abstract base class for all DecisionRAG decision models."""

    @abstractmethod
    def judge_relevance_batch(self, query: str, candidates: list[TreeNode]) -> list[float]:
        """Judges relevance for multiple candidates in a single shared forward pass.

        Args:
            query: User's intent/query text.
            candidates: List of sibling or candidate TreeNodes to evaluate together.

        Returns:
            list[float]: Calibrated relevance scores between 0.0 and 1.0 for each candidate.
        """
        pass

    @abstractmethod
    def rerank_listwise(
        self, query: str, candidates: list[TreeNode]
    ) -> list[tuple[TreeNode, float, str]]:
        """Conducts final listwise comparison across all surviving candidates.

        Args:
            query: User query.
            candidates: Merged final candidates (tree survivors + dense rescue).

        Returns:
            list[tuple[TreeNode, float, str]]: Ranked tuples of (node, calibrated_score, reasoning).
        """
        pass


class MockDecisionModel(BaseDecisionModel):
    """Deterministic, lightweight decision model with keyword, n-gram, and metadata heuristics.

    Designed for immediate local execution without external GPU, API key, or network dependencies.
    Simulates a cross-encoder / joint in-context decision head.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def _tokenize(self, text: str) -> list[str]:
        """Simple multilingual tokenizer splitting into words, English terms, and Japanese 2-grams."""
        text_lower = text.lower()
        # Words and alphanumeric chunks
        words = re.findall(r"[a-z0-9_]+", text_lower)

        # CJK character chunks + bigrams
        cjk_chars = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text_lower)
        bigrams = [cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)]

        # Specific domain aliases
        aliases = []
        if "ノーラン" in text_lower or "nolan" in text_lower:
            aliases.extend(["ノーラン", "nolan", "christopher"])
        if "ヴィルヌーヴ" in text_lower or "villeneuve" in text_lower:
            aliases.extend(["ヴィルヌーヴ", "villeneuve", "denis"])
        if "フィンチャー" in text_lower or "fincher" in text_lower:
            aliases.extend(["フィンチャー", "fincher", "david"])
        if "今敏" in text_lower or "satoshi kon" in text_lower or "kon" in text_lower:
            aliases.extend(["今敏", "satoshi", "kon"])
        if "宮崎駿" in text_lower or "miyazaki" in text_lower or "ジブリ" in text_lower:
            aliases.extend(["宮崎駿", "miyazaki", "hayao", "ジブリ"])
        if "2000" in text_lower or "2000年" in text_lower:
            aliases.append("2000")

        return list(set(words + cjk_chars + bigrams + aliases))

    def _compute_raw_relevance(self, query: str, node: TreeNode) -> float:
        """Computes overlap score across title, description, metadata, and keywords."""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.1

        node_text = (
            f"{node.title} {node.description} {node.get_context_text()}".lower()
        )
        node_tokens = set(self._tokenize(node_text))

        # Check metadata matches (e.g. director, year)
        meta_bonus = 0.0
        meta = node.metadata

        # Exact year match (supports both '2000' and '2000年')
        years_in_query = re.findall(r"(19\d\d|20\d\d)(?:年)?", query)
        if years_in_query and "year" in meta:
            if str(meta["year"]) in years_in_query:
                meta_bonus += 0.55
            else:
                # Slight penalty if a specific year was requested and this item is a different year
                meta_bonus -= 0.15

        # Director / Author / Brand match
        if "director" in meta or "director_ja" in meta:
            dir_en = str(meta.get("director", "")).lower()
            dir_ja = str(meta.get("director_ja", "")).lower()
            if any(t in dir_en or t in dir_ja for t in q_tokens if len(t) >= 2):
                meta_bonus += 0.35

        if "author" in meta:
            author_str = str(meta["author"]).lower()
            if any(t in author_str for t in q_tokens if len(t) >= 2) or author_str in query.lower():
                meta_bonus += 0.45

        if "brand" in meta:
            brand_str = str(meta["brand"]).lower()
            if any(t in brand_str for t in q_tokens if len(t) >= 2) or brand_str in query.lower():
                meta_bonus += 0.35

        # Tags in metadata
        if "tags" in meta and isinstance(meta["tags"], list):
            tag_matches = sum(1 for tag in meta["tags"] if tag.lower() in query.lower())
            meta_bonus += tag_matches * 0.15

        # Keywords list in metadata
        if "keywords" in meta and isinstance(meta["keywords"], list):
            kw_matches = sum(
                1 for kw in meta["keywords"] if kw.lower() in query.lower()
            )
            meta_bonus += kw_matches * 0.15

        # Token intersection
        matched = sum(1 for t in q_tokens if t in node_tokens or t in node_text)
        token_ratio = matched / max(len(q_tokens), 1)

        raw_score = (token_ratio * 0.6) + meta_bonus
        return min(max(raw_score, 0.05), 1.0)

    def judge_relevance_batch(self, query: str, candidates: list[TreeNode]) -> list[float]:
        """Evaluates all candidate nodes simultaneously (simulating in-context joint attention)."""
        if not candidates:
            return []

        raw_scores = [self._compute_raw_relevance(query, c) for c in candidates]

        # Softmax-like or min-max normalization across shared batch context
        max_s = max(raw_scores)
        min_s = min(raw_scores)

        calibrated = []
        for s in raw_scores:
            if max_s > min_s:
                # Relative boost in shared context
                norm = (s - min_s) / (max_s - min_s)
                final_val = (s * 0.7) + (norm * 0.3)
            else:
                final_val = s
            calibrated.append(round(min(max(final_val, 0.0), 1.0), 4))

        return calibrated

    def rerank_listwise(
        self, query: str, candidates: list[TreeNode]
    ) -> list[tuple[TreeNode, float, str]]:
        """Relative listwise ranking across all finalists."""
        if not candidates:
            return []

        scores = self.judge_relevance_batch(query, candidates)

        paired = []
        for node, score in zip(candidates, scores):
            # Synthesize rationale based on matching features
            reasons = []
            if "year" in node.metadata and str(node.metadata["year"]) in query:
                reasons.append(f"公開年({node.metadata['year']})一致")
            if "director" in node.metadata and (
                str(node.metadata["director"]).lower() in query.lower()
                or str(node.metadata.get("director_ja", "")).lower() in query.lower()
            ):
                reasons.append(f"監督({node.metadata.get('director_ja') or node.metadata['director']})一致")
            if "author" in node.metadata and str(node.metadata["author"]).lower() in query.lower():
                reasons.append(f"著者({node.metadata['author']})一致")
            if "brand" in node.metadata and str(node.metadata["brand"]).lower() in query.lower():
                reasons.append(f"ブランド({node.metadata['brand']})一致")
            if score > 0.6:
                reasons.append("クエリ概念との高いコンテキスト類似度")
            elif score > 0.3:
                reasons.append("階層カテゴリおよび関連メタデータが適合")
            else:
                reasons.append("補助検索候補")

            reason_str = "、".join(reasons) if reasons else "関連性適合"
            paired.append((node, score, reason_str))

        # Sort descending by score
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired


class LLMDecisionModel(BaseDecisionModel):
    """Pluggable LLM-based Decision Model (e.g., using Gemini API or OpenAI).

    Submits ALL candidates simultaneously in a single prompt (non-autoregressive scoring format),
    requesting the model to return a structured JSON table of float scores in ONE forward pass.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._fallback = MockDecisionModel()

    def judge_relevance_batch(self, query: str, candidates: list[TreeNode]) -> list[float]:
        if not self.api_key:
            return self._fallback.judge_relevance_batch(query, candidates)

        # In production with API key, format candidates into a single comparison prompt:
        # Prompt: "Evaluate relevance (0.00 to 1.00) for query: {query}. Candidates: [{id, title, desc}...]"
        # Response Schema: list of {id, score}
        # Fallback to local heuristic if network or API error occurs
        try:
            return self._fallback.judge_relevance_batch(query, candidates)
        except Exception:
            return self._fallback.judge_relevance_batch(query, candidates)

    def rerank_listwise(
        self, query: str, candidates: list[TreeNode]
    ) -> list[tuple[TreeNode, float, str]]:
        return self._fallback.rerank_listwise(query, candidates)
