"""Pydantic v2 schemas for DecisionRAG API requests, responses, candidates, and audit trace.

Supports Pydantic v2 when available with automatic dataclass fallback.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

try:
    from pydantic import BaseModel, Field

    class RecommendationRequest(BaseModel):
        """Search or recommendation query request payload."""

        query: str = Field(..., description="User query or context description", min_length=1)
        top_k: int = Field(default=5, description="Number of final ranked results to return", ge=1, le=20)
        branch_top_k: int = Field(
            default=2, description="Top-K child branches to preserve per parent level during pruning", ge=1, le=10
        )
        prune_threshold: float = Field(
            default=0.25, description="Minimum score threshold (0.0 - 1.0) required to survive pruning", ge=0.0, le=1.0
        )
        enable_dense_hybrid: bool = Field(
            default=True, description="Whether to merge dense embedding candidates into final pool"
        )
        dense_top_n: int = Field(default=3, description="Number of dense retrieval rescue candidates", ge=0, le=10)
        hybrid_alpha: float = Field(
            default=0.7, description="Tree vs Dense confidence weighting factor (0.0 to 1.0)", ge=0.0, le=1.0
        )
        auto_grow: bool = Field(
            default=False,
            description="Use Gemini to grow the knowledge tree when the best result scores below 0.30",
        )

    class EvaluatedNode(BaseModel):
        """Inspection of a single node evaluated in a forward pass."""

        id: str
        title: str
        level: int
        score: float
        survived: bool
        action: Literal["survived", "pruned", "rescued"] = "survived"
        reason: Optional[str] = None

    class DecisionTraceStep(BaseModel):
        """Detailed audit record of an exploration or fusion stage in DecisionRAG."""

        stage: str = Field(..., description="Stage name (e.g. 'Level 1: Category Scoring', 'Dense Candidate Fusion')")
        level: int = Field(default=0, description="Tree depth level")
        description: str = Field(..., description="Human-readable explanation of actions taken in this stage")
        evaluated: list[EvaluatedNode] = Field(default_factory=list)
        survivors_count: int = 0
        pruned_count: int = 0

    class CandidateResult(BaseModel):
        """Final output recommendation candidate."""

        rank: int = Field(..., description="Final listwise rank position (1-indexed)")
        id: str = Field(..., description="Node ID")
        title: str = Field(..., description="Work or Item Title")
        description: str = Field(..., description="Synopsis or summary text")
        level: int = Field(..., description="Hierarchy depth")
        score: float = Field(..., description="Final listwise calibrated relevance score (0.0 - 1.0)")
        source: Literal["tree_pruning", "dense_rescue", "both"] = Field(
            ..., description="Provenance of candidate: tree pruning survival, dense rescue, or both"
        )
        reasoning: str = Field(..., description="Model rationale for why this item was ranked here")
        breadcrumb: list[str] = Field(default_factory=list, description="Category path hierarchy titles")
        metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata attributes (director, year, keywords, etc.)")

    class RecommendationResponse(BaseModel):
        """Standard JSON response for /recommend and /search endpoints."""

        query: str
        results: list[CandidateResult]
        total_candidates_evaluated: int
        tree_survivors_count: int
        dense_rescued_count: int
        execution_time_ms: float
        trace: list[DecisionTraceStep]

    class DatasetInputSchema(BaseModel):
        """Schema for mounting a new dataset into the DecisionRAG tree."""

        id: Optional[str] = Field(default=None, description="Unique identifier for dataset")
        title: str = Field(..., description="Dataset or category title", min_length=1)
        description: str = Field(default="", description="Category description for decision model judgment")
        children: Optional[list[dict[str, Any]]] = Field(default_factory=list, description="Child nodes (recursive)")
        metadata: Optional[dict[str, Any]] = Field(default_factory=dict, description="Metadata dictionary")
        root_category: Optional[dict[str, Any]] = Field(default=None, description="Wrapped root category format")

    class DatasetInfo(BaseModel):
        """Metadata overview of a mounted dataset."""

        id: str
        title: str
        description: str
        is_default: bool
        enabled: bool
        node_count: int
        leaf_count: int

    class DatasetListResponse(BaseModel):
        datasets: list[DatasetInfo]

    class DatasetToggleRequest(BaseModel):
        enabled: bool

except ImportError:
    # Minimal dataclass fallback if pydantic is absent in execution environment
    from dataclasses import dataclass, field

    @dataclass
    class RecommendationRequest:  # type: ignore[no-redef]
        query: str
        top_k: int = 5
        branch_top_k: int = 2
        prune_threshold: float = 0.25
        enable_dense_hybrid: bool = True
        dense_top_n: int = 3
        hybrid_alpha: float = 0.7
        auto_grow: bool = False

    @dataclass
    class EvaluatedNode:  # type: ignore[no-redef]
        id: str
        title: str
        level: int
        score: float
        survived: bool
        action: str = "survived"
        reason: Optional[str] = None

        def to_dict(self) -> dict[str, Any]:
            return {
                "id": self.id,
                "title": self.title,
                "level": self.level,
                "score": self.score,
                "survived": self.survived,
                "action": self.action,
                "reason": self.reason,
            }

    @dataclass
    class DecisionTraceStep:  # type: ignore[no-redef]
        stage: str
        level: int = 0
        description: str = ""
        evaluated: list[Any] = field(default_factory=list)
        survivors_count: int = 0
        pruned_count: int = 0

        def to_dict(self) -> dict[str, Any]:
            return {
                "stage": self.stage,
                "level": self.level,
                "description": self.description,
                "evaluated": [
                    e.to_dict() if hasattr(e, "to_dict") else e for e in self.evaluated
                ],
                "survivors_count": self.survivors_count,
                "pruned_count": self.pruned_count,
            }

    @dataclass
    class CandidateResult:  # type: ignore[no-redef]
        rank: int
        id: str
        title: str
        description: str
        level: int
        score: float
        source: str
        reasoning: str
        breadcrumb: list[str] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return {
                "rank": self.rank,
                "id": self.id,
                "title": self.title,
                "description": self.description,
                "level": self.level,
                "score": self.score,
                "source": self.source,
                "reasoning": self.reasoning,
                "breadcrumb": self.breadcrumb,
                "metadata": self.metadata,
            }

    @dataclass
    class RecommendationResponse:  # type: ignore[no-redef]
        query: str
        results: list[Any]
        total_candidates_evaluated: int
        tree_survivors_count: int
        dense_rescued_count: int
        execution_time_ms: float
        trace: list[Any]

        def to_dict(self) -> dict[str, Any]:
            return {
                "query": self.query,
                "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.results],
                "total_candidates_evaluated": self.total_candidates_evaluated,
                "tree_survivors_count": self.tree_survivors_count,
                "dense_rescued_count": self.dense_rescued_count,
                "execution_time_ms": self.execution_time_ms,
                "trace": [t.to_dict() if hasattr(t, "to_dict") else t for t in self.trace],
            }

    @dataclass
    class DatasetInputSchema:  # type: ignore[no-redef]
        title: str
        id: Optional[str] = None
        description: str = ""
        children: list[Any] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)
        root_category: Optional[dict[str, Any]] = None

    @dataclass
    class DatasetInfo:  # type: ignore[no-redef]
        id: str
        title: str
        description: str
        is_default: bool
        enabled: bool
        node_count: int
        leaf_count: int

        def to_dict(self) -> dict[str, Any]:
            return {
                "id": self.id,
                "title": self.title,
                "description": self.description,
                "is_default": self.is_default,
                "enabled": self.enabled,
                "node_count": self.node_count,
                "leaf_count": self.leaf_count,
            }

    @dataclass
    class DatasetListResponse:  # type: ignore[no-redef]
        datasets: list[Any]

    @dataclass
    class DatasetToggleRequest:  # type: ignore[no-redef]
        enabled: bool
