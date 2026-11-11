"""FastAPI Application entry point for DecisionRAG.

Exposes endpoints:
- POST /recommend : Full DecisionRAG recommendation with tracing & parameters
- POST /search    : Simplified search query endpoint
- GET  /tree      : Complete hierarchical tree node index for visualization
- GET  /health    : System health & engine readiness
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings
from ..core.decision_model import MockDecisionModel, LLMDecisionModel
from ..core.dense_retriever import DenseRetriever
from ..core.engine import DecisionEngine
from ..core.tree import TreeIndex
from ..schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)

_global_engine: DecisionEngine | None = None


def get_engine() -> DecisionEngine:
    """Instantiates or retrieves the cached DecisionEngine singleton."""
    global _global_engine
    if _global_engine is None:
        settings = get_settings()
        catalog_path = Path(settings.data_path)

        # Fallback to local path if running from subfolder
        if not catalog_path.exists():
            alt_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_catalog.json"
            if alt_path.exists():
                catalog_path = alt_path

        if not catalog_path.exists():
            raise RuntimeError(f"Cannot locate catalog file at: {catalog_path}")

        tree_index = TreeIndex.from_file(catalog_path)

        # Select decision model backend
        if settings.decision_model_type == "llm" and settings.gemini_api_key:
            decision_model = LLMDecisionModel(api_key=settings.gemini_api_key)
        else:
            decision_model = MockDecisionModel()

        dense_retriever = DenseRetriever(use_sentence_transformers=False)

        _global_engine = DecisionEngine(
            tree_index=tree_index,
            decision_model=decision_model,
            dense_retriever=dense_retriever,
        )

    return _global_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up engine and indexes on startup."""
    get_engine()
    yield


app = FastAPI(
    title="DecisionRAG Recommendation & Search Engine",
    description=(
        "Production-ready boilerplate for Hierarchical Non-Generative DecisionRAG: "
        "combining hierarchical tree navigation, single-forward-pass batch decision pruning, "
        "dense candidate hybrid rescue, and listwise relative reranking."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Health check and index status."""
    engine = get_engine()
    leaves = engine.tree_index.get_all_leaves()
    return {
        "status": "healthy",
        "engine": "DecisionRAG",
        "total_nodes": len(engine.tree_index._node_map),
        "total_leaves": len(leaves),
        "root_title": engine.tree_index.root.title,
    }


@app.get("/tree", tags=["Catalog"])
def get_catalog_tree() -> dict[str, Any]:
    """Returns the full hierarchical catalog tree for visualization and exploration."""
    engine = get_engine()
    return engine.tree_index.root.to_dict()


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """Executes DecisionRAG search pipeline with custom hyperparameters and step-by-step trace."""
    engine = get_engine()
    try:
        return engine.search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=RecommendationResponse, tags=["Search"])
def search_simple(
    query: str = Query(..., description="Natural language search or recommendation query"),
    top_k: int = Query(5, ge=1, le=20),
    auto_grow: bool = Query(False, description="Grow the knowledge tree with Gemini on a knowledge gap"),
) -> RecommendationResponse:
    """Convenience endpoint for simple keyword/sentence queries with default pruning settings."""
    engine = get_engine()
    settings = get_settings()
    request = RecommendationRequest(
        query=query,
        top_k=top_k,
        branch_top_k=settings.branch_top_k,
        prune_threshold=settings.prune_threshold,
        enable_dense_hybrid=settings.enable_dense_hybrid,
        dense_top_n=settings.dense_top_n,
        auto_grow=auto_grow,
    )
    return engine.search(request)


# =============================================================================
# Dataset Management Endpoints
# =============================================================================

@app.get("/datasets", tags=["Dataset Management"])
def list_datasets() -> list[dict[str, Any]]:
    """Lists all currently mounted datasets with metadata and status."""
    engine = get_engine()
    return engine.list_datasets()


@app.post("/datasets", tags=["Dataset Management"])
def mount_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    """Mounts a new custom dataset into the hierarchical tree."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    # Validate minimal fields
    title = payload.get("title") or (payload.get("root_category", {}).get("title") if isinstance(payload.get("root_category"), dict) else None)
    if not title:
        raise HTTPException(status_code=422, detail="Missing required field: 'title'")

    engine = get_engine()
    try:
        info = engine.mount_dataset(payload)
        return {"status": "success", "dataset": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mount dataset: {str(e)}")


@app.delete("/datasets/{dataset_id}", tags=["Dataset Management"])
def unmount_dataset(dataset_id: str) -> dict[str, Any]:
    """Removes a dataset and its nodes completely from the tree and dense vector index."""
    engine = get_engine()
    success = engine.unmount_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return {"status": "success", "unmounted": dataset_id}


@app.patch("/datasets/{dataset_id}/toggle", tags=["Dataset Management"])
def toggle_dataset(dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Enables or disables a dataset from search exploration without deleting it."""
    enabled = payload.get("enabled", True)
    engine = get_engine()
    success = engine.toggle_dataset(dataset_id, enabled)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return {"status": "success", "dataset_id": dataset_id, "enabled": enabled}


@app.get("/datasets/presets", tags=["Dataset Management"])
def get_dataset_presets() -> list[dict[str, Any]]:
    """Returns preset dataset templates (e.g. Ochiai Yoichi books, EC gadgets)."""
    presets = []
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"

    ochiai_file = base_dir / "sample_books_ochiai.json"
    if ochiai_file.exists():
        import json
        with open(ochiai_file, "r", encoding="utf-8") as f:
            presets.append({
                "preset_id": "ochiai_books",
                "label": "📚 落合陽一 著作・デジタル思想カタログ",
                "description": "『デジタルネイチャー』『魔法の世紀』『ビヨンド・デジタル・ネイチャー』を含む書籍ツリー",
                "data": json.load(f),
            })

    ecommerce_file = base_dir / "sample_ecommerce.json"
    if ecommerce_file.exists():
        import json
        with open(ecommerce_file, "r", encoding="utf-8") as f:
            presets.append({
                "preset_id": "ecommerce_gadgets",
                "label": "🛍️ EC商品・スマート家電＆ガジェット",
                "description": "Sony WH-1000XM5等のオーディオ・スマート機器カタログ",
                "data": json.load(f),
            })

    return presets


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
