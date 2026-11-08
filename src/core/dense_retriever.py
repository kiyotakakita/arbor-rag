"""Dense Retriever for candidate rescue in DecisionRAG.

Provides semantic embedding retrieval across all leaf/item passages.
Supports Sentence-Transformers when installed, and provides a built-in
lightweight TF-IDF / character n-gram cosine vectorizer fallback that works
without any external packages.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .tree import TreeNode


class DenseRetriever:
    """Vector-based dense retriever to prevent pruning omissions."""

    def __init__(self, use_sentence_transformers: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.use_sentence_transformers = use_sentence_transformers
        self.model_name = model_name
        self.nodes: list[TreeNode] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._st_model = None

        if use_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(model_name)
            except ImportError:
                self._st_model = None

    def _tokenize(self, text: str) -> list[str]:
        text_lower = text.lower()
        words = re.findall(r"[a-z0-9_]+", text_lower)
        cjk = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text_lower)
        bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return words + cjk + bigrams

    def _text_to_vector(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        vec: dict[str, float] = {}
        total = len(tokens)
        for term, count in counts.items():
            tf = count / total
            idf = self._idf.get(term, 1.0)
            vec[term] = tf * idf

        # Normalize L2
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for term in vec:
                vec[term] /= norm
        return vec

    def _cosine_similarity(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        # Dot product
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        return sum(vec_a[k] * vec_b[k] for k in common_keys)

    def index_nodes(self, nodes: list[TreeNode]) -> None:
        """Indexes candidate leaf nodes into the vector space."""
        self.nodes = nodes
        corpus_docs = [n.get_context_text() for n in nodes]

        if self._st_model is not None:
            self._dense_embeddings = self._st_model.encode(corpus_docs)
            return

        # Built-in TF-IDF inverted index
        doc_count = len(corpus_docs)
        doc_freq: Counter[str] = Counter()

        tokenized_docs = [self._tokenize(doc) for doc in corpus_docs]
        for tokens in tokenized_docs:
            for term in set(tokens):
                doc_freq[term] += 1

        self._idf = {
            term: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0
            for term, freq in doc_freq.items()
        }

        self._vectors = [self._text_to_vector(doc) for doc in corpus_docs]

    def update_nodes(self, nodes: list[TreeNode]) -> None:
        """Dynamically re-indexes the vector space with currently active leaf nodes."""
        self.index_nodes(nodes)

    def retrieve(self, query: str, top_n: int = 3) -> list[tuple[TreeNode, float]]:
        """Retrieves Top-N candidate nodes matching the query via dense/embedding similarity."""
        if not self.nodes:
            return []

        if self._st_model is not None:
            import numpy as np
            q_emb = self._st_model.encode([query])[0]
            scores = np.dot(self._dense_embeddings, q_emb) / (
                np.linalg.norm(self._dense_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
            )
            ranked_indices = np.argsort(scores)[::-1][:top_n]
            return [(self.nodes[i], float(scores[i])) for i in ranked_indices]

        # Built-in cosine retriever
        q_vec = self._text_to_vector(query)
        scored: list[tuple[TreeNode, float]] = []
        for node, doc_vec in zip(self.nodes, self._vectors):
            sim = self._cosine_similarity(q_vec, doc_vec)
            scored.append((node, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
