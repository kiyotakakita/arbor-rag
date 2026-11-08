#!/usr/bin/env python3
"""JSON output CLI utility for DecisionRAG.

Allows backend bridge or scripts to query DecisionRAG and receive structured JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.decision_model import MockDecisionModel
from src.core.dense_retriever import DenseRetriever
from src.core.engine import DecisionEngine
from src.core.tree import TreeIndex
from src.schemas.recommendation import RecommendationRequest


def main():
    catalog_path = Path(__file__).resolve().parent.parent / "data" / "sample_catalog.json"
    if not catalog_path.exists():
        print(json.dumps({"error": f"Catalog not found: {catalog_path}"}), file=sys.stderr)
        sys.exit(1)

    tree = TreeIndex.from_file(catalog_path)

    # Load any persisted custom datasets in data/datasets/
    datasets_dir = Path(__file__).resolve().parent.parent / "data" / "datasets"
    if datasets_dir.exists():
        for custom_file in sorted(datasets_dir.glob("*.json")):
            try:
                with open(custom_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    tree.mount_dataset(cdata)
            except Exception as e:
                print(f"Warning: Failed to load {custom_file}: {e}", file=sys.stderr)

    engine = DecisionEngine(
        tree_index=tree,
        decision_model=MockDecisionModel(),
        dense_retriever=DenseRetriever(),
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", action="store_true", help="Print catalog tree")
    parser.add_argument("--datasets", action="store_true", help="List mounted datasets")
    parser.add_argument("--stdin", action="store_true", help="Read JSON request from stdin")
    parser.add_argument("--query", type=str, default="", help="Query string")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--branch_top_k", type=int, default=2)
    parser.add_argument("--prune_threshold", type=float, default=0.25)
    parser.add_argument("--enable_dense_hybrid", type=str, default="true")
    parser.add_argument("--dense_top_n", type=int, default=3)
    parser.add_argument("raw_query", nargs="*", default=[])
    parser.add_argument("--gardener-grow", type=str, default="", help="Run Gardener auto-graft on query")
    parser.add_argument("--gardener-branch-json", type=str, default="", help="Optional JSON string of custom branch to graft")
    args = parser.parse_args()

    if args.tree:
        print(json.dumps(tree.root.to_dict(), ensure_ascii=False))
        return

    if args.datasets:
        print(json.dumps(tree.list_datasets(), ensure_ascii=False))
        return

    if args.gardener_grow:
        from src.core.gardener import GardenerWorker
        gardener = GardenerWorker()
        custom_branch = None
        if args.gardener_branch_json:
            try:
                custom_branch = json.loads(args.gardener_branch_json)
            except Exception:
                pass

        event, after_res = gardener.auto_graft_and_evolve(engine, args.gardener_grow, custom_branch)

        # Persist grafted branch to data/datasets/
        if event.grafted_dataset_id:
            grafted_node = tree.get_node(event.grafted_dataset_id)
            if grafted_node:
                target_file = datasets_dir / f"{event.grafted_dataset_id}.json"
                try:
                    with open(target_file, "w", encoding="utf-8") as f:
                        json.dump(grafted_node.to_dict(), f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to persist grafted file: {e}", file=sys.stderr)

        out = {
            "event": event.to_dict(),
            "response": after_res.to_dict() if hasattr(after_res, "to_dict") else after_res.model_dump(),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.stdin:
        input_data = sys.stdin.read().strip()
        data = json.loads(input_data) if input_data else {}
        req = RecommendationRequest(
            query=data.get("query", "ノーラン監督の2000年の映画"),
            top_k=data.get("top_k", 5),
            branch_top_k=data.get("branch_top_k", 2),
            prune_threshold=data.get("prune_threshold", 0.25),
            enable_dense_hybrid=data.get("enable_dense_hybrid", True),
            dense_top_n=data.get("dense_top_n", 3),
            hybrid_alpha=data.get("hybrid_alpha", 0.7),
        )
    else:
        q = args.query or (" ".join(args.raw_query) if args.raw_query else "ノーラン監督の2000年の映画")
        req = RecommendationRequest(
            query=q,
            top_k=args.top_k,
            branch_top_k=args.branch_top_k,
            prune_threshold=args.prune_threshold,
            enable_dense_hybrid=args.enable_dense_hybrid.lower() in ("true", "1"),
            dense_top_n=args.dense_top_n,
        )

    response = engine.search(req)

    # Convert to json-serializable dict
    if hasattr(response, "to_dict"):
        out = response.to_dict()
    elif hasattr(response, "model_dump"):
        out = response.model_dump()
    else:
        out = {
            "query": response.query,
            "results": [r.__dict__ for r in response.results],
            "total_candidates_evaluated": response.total_candidates_evaluated,
            "tree_survivors_count": response.tree_survivors_count,
            "dense_rescued_count": response.dense_rescued_count,
            "execution_time_ms": response.execution_time_ms,
            "trace": [t.__dict__ for t in response.trace],
        }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
