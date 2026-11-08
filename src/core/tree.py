"""Hierarchical tree representation and dynamic indexing utilities for DecisionRAG.

Supports:
- Arbitrary tree depth (Corpus -> Category -> Subcategory -> Item -> Detail/Passage)
- Dynamic mounting & unmounting of datasets from UI/API
- Enabled/disabled dataset toggling without data loss
- Active leaf aggregation for dense retrieval synchronization
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TreeNode:
    """Represents a node in the hierarchical navigation tree."""

    id: str
    title: str
    description: str
    level: int = 0
    children: list[TreeNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    enabled: bool = True
    dataset_id: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        """Returns True if the node has no children (e.g. item or passage leaf)."""
        return len(self.children) == 0

    def get_context_text(self) -> str:
        """Constructs a comprehensive representation text for scoring & dense retrieval."""
        parts = [self.title, self.description]

        # Extract useful metadata keywords/director/author/year/tags if present
        if self.metadata:
            meta_strings = []
            for k, v in self.metadata.items():
                if isinstance(v, list):
                    meta_strings.append(f"{k}: {', '.join(str(item) for item in v)}")
                elif isinstance(v, (str, int, float)):
                    meta_strings.append(f"{k}: {v}")
            if meta_strings:
                parts.append(" | ".join(meta_strings))

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serializes node to a clean dictionary structure."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "level": self.level,
            "metadata": self.metadata,
            "enabled": self.enabled,
            "dataset_id": self.dataset_id,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        parent_id: Optional[str] = None,
        level: Optional[int] = None,
        dataset_id: Optional[str] = None,
    ) -> TreeNode:
        """Recursively parses a dictionary into a TreeNode tree."""
        node_id = str(data.get("id") or f"node_{abs(hash(data.get('title', 'untitled')))}")
        node_level = level if level is not None else data.get("level", 0)
        curr_dataset_id = dataset_id or data.get("dataset_id")

        node = cls(
            id=node_id,
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            level=node_level,
            metadata=data.get("metadata", {}),
            parent_id=parent_id,
            enabled=data.get("enabled", True),
            dataset_id=curr_dataset_id,
        )

        children_data = data.get("children", [])
        node.children = [
            cls.from_dict(
                c,
                parent_id=node.id,
                level=node_level + 1,
                dataset_id=curr_dataset_id,
            )
            for c in children_data
        ]
        return node


class TreeIndex:
    """Index structure for fast traversal, lookup, dynamic mounting, and active leaf aggregation."""

    def __init__(self, root: TreeNode):
        self.root = root
        self._node_map: dict[str, TreeNode] = {}
        self._level_map: dict[int, list[TreeNode]] = {}
        self._leaves: list[TreeNode] = []
        self._parent_map: dict[str, str] = {}
        self._mounted_datasets: dict[str, dict[str, Any]] = {}
        self._rebuild_all_indexes()

    def _rebuild_all_indexes(self) -> None:
        """Completely rebuilds all node lookup maps and leaf lists."""
        self._node_map.clear()
        self._level_map.clear()
        self._leaves.clear()
        self._parent_map.clear()

        # Build from root
        self._index_subtree(self.root, parent_id=None)

        # Register default dataset if not tracked yet
        if "default_movies" not in self._mounted_datasets:
            movie_children = [c for c in self.root.children if not c.dataset_id or c.dataset_id == "default_movies"]
            if movie_children:
                total_nodes = sum(self._count_nodes(c) for c in movie_children)
                total_leaves = sum(len(self._collect_leaves_of(c)) for c in movie_children)
                self._mounted_datasets["default_movies"] = {
                    "id": "default_movies",
                    "title": "映画カタログ（デフォルト）",
                    "description": "SF・サスペンス・アニメーションの階層化映画知識ベース",
                    "is_default": True,
                    "enabled": True,
                    "node_count": total_nodes,
                    "leaf_count": total_leaves,
                    "root_node_ids": [c.id for c in movie_children],
                }

    def _index_subtree(self, current: TreeNode, parent_id: Optional[str] = None) -> None:
        self._node_map[current.id] = current

        if parent_id is not None:
            self._parent_map[current.id] = parent_id
            current.parent_id = parent_id

        if current.level not in self._level_map:
            self._level_map[current.level] = []
        self._level_map[current.level].append(current)

        if current.is_leaf:
            self._leaves.append(current)

        for child in current.children:
            self._index_subtree(child, current.id)

    def _count_nodes(self, node: TreeNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _collect_leaves_of(self, node: TreeNode) -> list[TreeNode]:
        if node.is_leaf:
            return [node]
        leaves: list[TreeNode] = []
        for c in node.children:
            leaves.extend(self._collect_leaves_of(c))
        return leaves

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self._node_map.get(node_id)

    def get_parent(self, node_id: str) -> Optional[TreeNode]:
        parent_id = self._parent_map.get(node_id)
        return self._node_map.get(parent_id) if parent_id else None

    def get_path_to_node(self, node_id: str) -> list[TreeNode]:
        path: list[TreeNode] = []
        curr = self.get_node(node_id)
        while curr:
            path.append(curr)
            curr = self.get_parent(curr.id)
        path.reverse()
        return path

    def get_nodes_at_level(self, level: int) -> list[TreeNode]:
        return self._level_map.get(level, [])

    def get_all_leaves(self) -> list[TreeNode]:
        return self._leaves

    def get_active_leaves(self) -> list[TreeNode]:
        """Returns all leaf nodes whose top-level dataset / branch is currently enabled."""
        return [l for l in self._leaves if self.is_node_enabled(l.id)]

    def is_node_enabled(self, node_id: str) -> bool:
        """Checks if a node is enabled and whether all its ancestors are enabled."""
        curr = self.get_node(node_id)
        while curr:
            if not getattr(curr, "enabled", True):
                return False
            curr = self.get_parent(curr.id)
        return True

    def get_item_or_leaf_parent(self, node: TreeNode) -> TreeNode:
        if node.level > 3 and node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent:
                return parent
        return node

    # =========================================================================
    # Dynamic Dataset Management (CRUD)
    # =========================================================================

    def mount_dataset(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mounts a new dataset into the tree under Root (Level 1).

        Supports both standard TreeNode format and wrapped format:
        - Format A: { "id": "...", "title": "...", "description": "...", "children": [...] }
        - Format B: { "id": "...", "title": "...", "root_category": { ... } }
        """
        dataset_id = str(data.get("id") or f"dataset_{abs(hash(data.get('title', 'dataset')))}")

        # Check if already mounted
        if dataset_id in self._mounted_datasets:
            self.unmount_dataset(dataset_id)

        # Handle wrapped format or direct format
        if "root_category" in data and isinstance(data["root_category"], dict):
            cat_data = dict(data["root_category"])
            cat_data.setdefault("id", dataset_id)
            cat_data.setdefault("title", data.get("title", cat_data.get("title", "新規データセット")))
            cat_data.setdefault("description", data.get("description", cat_data.get("description", "")))
        else:
            cat_data = dict(data)
            cat_data["id"] = dataset_id

        # Instantiate tree node as Level 1 branch under root
        branch_node = TreeNode.from_dict(
            cat_data,
            parent_id=self.root.id,
            level=1,
            dataset_id=dataset_id,
        )
        branch_node.enabled = True

        # Append to root's children
        self.root.children.append(branch_node)

        # Re-index all structures
        self._rebuild_all_indexes()

        node_count = self._count_nodes(branch_node)
        leaf_count = len(self._collect_leaves_of(branch_node))

        dataset_info = {
            "id": dataset_id,
            "title": branch_node.title,
            "description": branch_node.description,
            "is_default": False,
            "enabled": True,
            "node_count": node_count,
            "leaf_count": leaf_count,
            "root_node_ids": [branch_node.id],
        }
        self._mounted_datasets[dataset_id] = dataset_info
        return dataset_info

    def unmount_dataset(self, dataset_id: str) -> bool:
        """Removes a dataset branch completely from root and all lookup indexes."""
        if dataset_id not in self._mounted_datasets:
            # Check if matching root child ID directly
            matching = [c for c in self.root.children if c.id == dataset_id or c.dataset_id == dataset_id]
            if not matching:
                return False

        # Filter out from root children
        self.root.children = [
            c for c in self.root.children if c.id != dataset_id and c.dataset_id != dataset_id
        ]

        if dataset_id in self._mounted_datasets:
            del self._mounted_datasets[dataset_id]

        self._rebuild_all_indexes()
        return True

    def toggle_dataset(self, dataset_id: str, enabled: bool) -> bool:
        """Enables or disables a dataset from search exploration without removing it."""
        if dataset_id in self._mounted_datasets:
            self._mounted_datasets[dataset_id]["enabled"] = enabled

        # Update root child matching this dataset
        found = False
        for c in self.root.children:
            if c.id == dataset_id or c.dataset_id == dataset_id or (
                dataset_id == "default_movies" and (not c.dataset_id or c.dataset_id == "default_movies")
            ):
                c.enabled = enabled
                self._set_enabled_recursive(c, enabled)
                found = True

        return found

    def _set_enabled_recursive(self, node: TreeNode, enabled: bool) -> None:
        node.enabled = enabled
        for c in node.children:
            self._set_enabled_recursive(c, enabled)

    def list_datasets(self) -> list[dict[str, Any]]:
        """Returns metadata list of all mounted datasets."""
        # Refresh counts
        result = []
        for ds_id, info in self._mounted_datasets.items():
            # calculate active node count
            root_children = [
                c for c in self.root.children
                if c.dataset_id == ds_id or c.id == ds_id or (ds_id == "default_movies" and not c.dataset_id)
            ]
            total_nodes = sum(self._count_nodes(c) for c in root_children)
            total_leaves = sum(len(self._collect_leaves_of(c)) for c in root_children)
            is_enabled = any(c.enabled for c in root_children) if root_children else False

            result.append({
                "id": ds_id,
                "title": info.get("title", ds_id),
                "description": info.get("description", ""),
                "is_default": info.get("is_default", False),
                "enabled": is_enabled,
                "node_count": total_nodes,
                "leaf_count": total_leaves,
            })
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> TreeIndex:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        root = TreeNode.from_dict(data)
        return cls(root)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreeIndex:
        root = TreeNode.from_dict(data)
        return cls(root)
