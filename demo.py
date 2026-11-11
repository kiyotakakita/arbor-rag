#!/usr/bin/env python3
"""Interactive CLI demo for DecisionRAG.

Usage:
    python3 demo.py
    python3 demo.py --query "ノーラン監督の2000年の映画"
    python3 demo.py --query "夢の中に潜入する産業スパイ"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core.decision_model import MockDecisionModel
from src.core.dense_retriever import DenseRetriever
from src.core.engine import DecisionEngine
from src.core.tree import TreeIndex
from src.schemas.recommendation import RecommendationRequest


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="DecisionRAG Hierarchical Search & Decision CLI")
    parser.add_argument(
        "--query",
        type=str,
        default="ノーラン監督の2000年の映画",
        help="Query to evaluate",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Final top K items")
    parser.add_argument("--branch_k", type=int, default=2, help="Branch factor Top-K per level")
    parser.add_argument("--prune_thresh", type=float, default=0.2, help="Pruning threshold")
    parser.add_argument("--dense_n", type=int, default=3, help="Dense hybrid candidates count")
    parser.add_argument(
        "--auto-grow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically grow the knowledge tree with Gemini on a knowledge gap",
    )
    args = parser.parse_args()

    catalog_path = Path(__file__).resolve().parent / "data" / "sample_catalog.json"
    if not catalog_path.exists():
        print(f"Error: Catalog not found at {catalog_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("🌲 DecisionRAG Hierarchical Decision & Recommendation Engine")
    print("=" * 70)
    print(f"🔍 Query: \"{args.query}\"")
    print(
        f"⚙️  Settings: branch_top_k={args.branch_k}, "
        f"prune_threshold={args.prune_thresh}, dense_top_n={args.dense_n}, "
        f"auto_grow={args.auto_grow}"
    )
    print("-" * 70)

    # Initialize Engine
    tree = TreeIndex.from_file(catalog_path)
    engine = DecisionEngine(
        tree_index=tree,
        decision_model=MockDecisionModel(),
        dense_retriever=DenseRetriever(),
    )

    req = RecommendationRequest(
        query=args.query,
        top_k=args.top_k,
        branch_top_k=args.branch_k,
        prune_threshold=args.prune_thresh,
        enable_dense_hybrid=True,
        dense_top_n=args.dense_n,
        auto_grow=args.auto_grow,
    )

    response = engine.search(req)

    print("\n📍 EXPLORATION & DECISION TRACE (階層枝刈り & 決定プロセス):")
    for step_idx, step in enumerate(response.trace, 1):
        print(f"\n[Step {step_idx}] {step.stage}")
        print(f"  ℹ️  {step.description}")
        for eval_node in step.evaluated:
            symbol = "✅ [SURVIVED]" if eval_node.survived else "✂️  [PRUNED]"
            if eval_node.action == "rescued":
                symbol = "🛟 [RESCUED]"
            print(f"     {symbol} {eval_node.title:<32} (Score: {eval_node.score:.3f}) -> {eval_node.reason}")

    print("\n" + "=" * 70)
    print(f"🏆 FINAL LISTWISE RANKED RESULTS (最終順位決定 / {response.execution_time_ms}ms):")
    print("=" * 70)
    for res in response.results:
        source_label = (
            "🌿 Tree Pruning"
            if res.source == "tree_pruning"
            else ("🛟 Dense Rescue" if res.source == "dense_rescue" else "⭐ Tree + Dense")
        )
        print(f"\n#{res.rank}  {res.title}  [Score: {res.score:.3f} | {source_label}]")
        print(f"    Path: {' ➔ '.join(res.breadcrumb)}")
        if res.metadata:
            meta_parts = [f"{k}: {v}" for k, v in res.metadata.items() if k not in ("cast",)]
            print(f"    Meta: {', '.join(meta_parts)}")
        print(f"    Why:  {res.reasoning}")
        print(f"    Desc: {res.description}")

    print("\n" + "=" * 70)
    print("✨ Complete. Ready to deploy via FastAPI (`uvicorn src.api.app:app --reload`)")
    print("=" * 70)


if __name__ == "__main__":
    main()
