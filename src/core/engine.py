"""DecisionRAG Engine: Core orchestrator for hierarchical tree navigation,
batch forward-pass pruning, dense candidate rescue, and listwise reranking.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from ..schemas.recommendation import (
    CandidateResult,
    DecisionTraceStep,
    EvaluatedNode,
    RecommendationRequest,
    RecommendationResponse,
)
from .decision_model import BaseDecisionModel, MockDecisionModel
from .dense_retriever import DenseRetriever
from .gardener import TreeGardener
from .tree import TreeIndex, TreeNode

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Orchestrates the DecisionRAG search and recommendation workflow."""

    def __init__(
        self,
        tree_index: TreeIndex,
        decision_model: Optional[BaseDecisionModel] = None,
        dense_retriever: Optional[DenseRetriever] = None,
    ):
        self.tree_index = tree_index
        self.decision_model = decision_model or MockDecisionModel()
        self.dense_retriever = dense_retriever or DenseRetriever()

        # Pre-index all active leaf items into the dense retriever
        self._sync_dense_index()

    def _sync_dense_index(self) -> None:
        """Indexes currently active leaf nodes into the vector search space."""
        active_leaves = self.tree_index.get_active_leaves()
        self.dense_retriever.update_nodes(active_leaves)

    def mount_dataset(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mounts a dataset and syncs the dense vector index."""
        info = self.tree_index.mount_dataset(data)
        self._sync_dense_index()
        return info

    def unmount_dataset(self, dataset_id: str) -> bool:
        """Unmounts a dataset and removes its vectors from dense retrieval."""
        res = self.tree_index.unmount_dataset(dataset_id)
        self._sync_dense_index()
        return res

    def toggle_dataset(self, dataset_id: str, enabled: bool) -> bool:
        """Toggles dataset status and updates active dense index."""
        res = self.tree_index.toggle_dataset(dataset_id, enabled)
        self._sync_dense_index()
        return res

    def list_datasets(self) -> list[dict[str, Any]]:
        """Returns metadata list of all mounted datasets."""
        return self.tree_index.list_datasets()

    def search(
        self, request: RecommendationRequest, auto_grow: bool | None = None
    ) -> RecommendationResponse:
        """Executes the complete DecisionRAG hierarchical retrieval & listwise ranking pipeline."""
        start_time = time.perf_counter()
        should_auto_grow = request.auto_grow if auto_grow is None else auto_grow
        trace: list[DecisionTraceStep] = []
        total_evaluated_count = 0

        # Step 1: Initialize candidates from Root's active (enabled) children (Level 1 Categories)
        current_active_nodes = [
            c for c in self.tree_index.root.children if getattr(c, "enabled", True)
        ]
        current_level = 1

        # Tree navigation loop across hierarchy depths
        while current_active_nodes:
            if not current_active_nodes:
                break

            total_evaluated_count += len(current_active_nodes)

            # Single forward-pass batch scoring across current siblings
            scores = self.decision_model.judge_relevance_batch(
                request.query, current_active_nodes
            )

            # Pair nodes with scores
            scored_candidates = list(zip(current_active_nodes, scores))
            # Sort by score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Pruning logic: Retain top branch_top_k that exceed prune_threshold
            survivors: list[TreeNode] = []
            evaluated_step_nodes: list[EvaluatedNode] = []

            for rank_idx, (node, score) in enumerate(scored_candidates):
                survived = (rank_idx < request.branch_top_k) and (
                    score >= request.prune_threshold or rank_idx == 0
                )
                action = "survived" if survived else "pruned"
                reason = (
                    f"Score {score:.2f} >= {request.prune_threshold} (Top {rank_idx+1})"
                    if survived
                    else (
                        f"Pruned: Below threshold {request.prune_threshold}"
                        if score < request.prune_threshold
                        else f"Pruned: Exceeded branch Top-{request.branch_top_k}"
                    )
                )

                evaluated_step_nodes.append(
                    EvaluatedNode(
                        id=node.id,
                        title=node.title,
                        level=node.level,
                        score=score,
                        survived=survived,
                        action=action,
                        reason=reason,
                    )
                )

                if survived:
                    survivors.append(node)

            trace.append(
                DecisionTraceStep(
                    stage=f"Level {current_level} Hierarchical Navigation & Batch Pruning",
                    level=current_level,
                    description=(
                        f"Evaluated {len(current_active_nodes)} candidate branches in a single shared forward pass. "
                        f"Preserved top {len(survivors)} branches, pruned {len(current_active_nodes) - len(survivors)}."
                    ),
                    evaluated=evaluated_step_nodes,
                    survivors_count=len(survivors),
                    pruned_count=len(current_active_nodes) - len(survivors),
                )
            )

            # Check if all survivors are leaf nodes or if we need to expand children
            has_children = any(len(n.children) > 0 for n in survivors)
            if not has_children:
                # Reached terminal depth
                current_active_nodes = survivors
                break

            # Expand enabled children for next level
            next_level_nodes: list[TreeNode] = []
            for s in survivors:
                active_children = [c for c in s.children if getattr(c, "enabled", True)]
                if active_children:
                    next_level_nodes.extend(active_children)
                else:
                    # Leaf node that finished earlier than siblings
                    next_level_nodes.append(s)

            current_active_nodes = next_level_nodes
            current_level += 1

        # Collect leaf / target item representations from survivors
        tree_survivor_items: list[TreeNode] = []
        for n in current_active_nodes:
            # Map passage level 4 back to parent item if preferred, or keep leaf
            target = self.tree_index.get_item_or_leaf_parent(n)
            if target not in tree_survivor_items:
                tree_survivor_items.append(target)

        tree_item_ids = {n.id for n in tree_survivor_items}

        # Step 4: Dense Retrieval Candidate Rescue (Hybrid Fusion)
        dense_rescued_items: list[TreeNode] = []
        dense_trace_nodes: list[EvaluatedNode] = []

        if request.enable_dense_hybrid and request.dense_top_n > 0:
            dense_results = self.dense_retriever.retrieve(
                request.query, top_n=request.dense_top_n
            )

            for leaf_node, sim_score in dense_results:
                target_item = self.tree_index.get_item_or_leaf_parent(leaf_node)
                is_rescue = target_item.id not in tree_item_ids

                dense_trace_nodes.append(
                    EvaluatedNode(
                        id=target_item.id,
                        title=target_item.title,
                        level=target_item.level,
                        score=sim_score,
                        survived=True,
                        action="rescued" if is_rescue else "survived",
                        reason=(
                            f"Rescued via Dense Retrieval (Cosine Sim: {sim_score:.3f})"
                            if is_rescue
                            else f"Corroborated by Dense Retrieval (Cosine Sim: {sim_score:.3f})"
                        ),
                    )
                )

                if is_rescue and target_item not in dense_rescued_items:
                    dense_rescued_items.append(target_item)

            trace.append(
                DecisionTraceStep(
                    stage="Dense Candidates Rescue (Hybrid Fallback)",
                    level=current_level,
                    description=(
                        f"Retrieved Top-{request.dense_top_n} dense candidates via vector embedding space. "
                        f"Rescued {len(dense_rescued_items)} candidates that were bypassed or pruned during hierarchical search."
                    ),
                    evaluated=dense_trace_nodes,
                    survivors_count=len(dense_trace_nodes),
                    pruned_count=0,
                )
            )

        # Step 5: Merge and Deduplicate candidates
        candidate_sources: dict[str, str] = {}
        merged_candidates: list[TreeNode] = []

        for item in tree_survivor_items:
            candidate_sources[item.id] = "tree_pruning"
            merged_candidates.append(item)

        for item in dense_rescued_items:
            if item.id in candidate_sources:
                candidate_sources[item.id] = "both"
            else:
                candidate_sources[item.id] = "dense_rescue"
                merged_candidates.append(item)

        # Fallback safeguard: if tree was over-pruned and no dense candidates, use level 1 top items
        if not merged_candidates:
            all_leaves = self.tree_index.get_all_leaves()[: request.top_k]
            merged_candidates = [
                self.tree_index.get_item_or_leaf_parent(l) for l in all_leaves
            ]
            for item in merged_candidates:
                candidate_sources[item.id] = "dense_rescue"

        # Step 6: Listwise Final Ranking
        # Evaluates all remaining finalists in a shared mutual comparison context
        ranked_tuples = self.decision_model.rerank_listwise(
            request.query, merged_candidates
        )

        final_results: list[CandidateResult] = []
        for rank, (node, score, reasoning) in enumerate(
            ranked_tuples[: request.top_k], start=1
        ):
            # Compute breadcrumb path
            path = self.tree_index.get_path_to_node(node.id)
            breadcrumb = [p.title for p in path if p.level > 0]

            final_results.append(
                CandidateResult(
                    rank=rank,
                    id=node.id,
                    title=node.title,
                    description=node.description,
                    level=node.level,
                    score=round(score, 4),
                    source=candidate_sources.get(node.id, "tree_pruning"),  # type: ignore[arg-type]
                    reasoning=reasoning,
                    breadcrumb=breadcrumb,
                    metadata=node.metadata,
                )
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        response = RecommendationResponse(
            query=request.query,
            results=final_results,
            total_candidates_evaluated=total_evaluated_count + len(dense_rescued_items),
            tree_survivors_count=len(tree_survivor_items),
            dense_rescued_count=len(dense_rescued_items),
            execution_time_ms=elapsed_ms,
            trace=trace,
        )

        top_score_before = response.results[0].score if response.results else 0.0
        if should_auto_grow and top_score_before < 0.50:
            top_title_before = response.results[0].title if response.results else "該当なし"
            logger.info("🌱 庭師エージェントが発火: Gemini に問い合わせ中...")
            branch = TreeGardener().cultivate(request.query)
            mount_info = self.mount_dataset(branch)
            leaf_titles = [
                node.title
                for node in self.tree_index.get_active_leaves()
                if node.dataset_id == mount_info["id"]
            ]
            mounted_label = f"{mount_info['title']} ➔ {', '.join(leaf_titles[:3])}"
            logger.info(
                "✅ 新規ツリー接ぎ木完了: 【%s】 (%d leaf items)",
                mounted_label,
                mount_info["leaf_count"],
            )
            evolved_response = self.search(request, auto_grow=False)
            top_score_after = (
                evolved_response.results[0].score if evolved_response.results else 0.0
            )
            top_title_after = (
                evolved_response.results[0].title
                if evolved_response.results
                else "該当なし"
            )
            evolved_response.trace.insert(
                0,
                DecisionTraceStep(
                    stage="🌱 [GARDENER AUTO-MOUNT]",
                    level=1,
                    description=(
                        f"Geminiにより「{mounted_label}」のツリーを接ぎ木しました。"
                        f"【進化前】{top_title_before} (Score: {top_score_before:.3f}) → "
                        f"【自己進化後】{top_title_after} (Score: {top_score_after:.3f})"
                    ),
                    evaluated=[],
                    survivors_count=mount_info["leaf_count"],
                    pruned_count=0,
                ),
            )
            evolved_response.execution_time_ms = round(
                (time.perf_counter() - start_time) * 1000.0, 2
            )
            logger.info(
                "🌱 自己進化後: %s (Score: %.3f)",
                top_title_after,
                top_score_after,
            )
            return evolved_response

        return response
