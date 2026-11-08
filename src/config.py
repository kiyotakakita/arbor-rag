"""Configuration management for DecisionRAG.

Supports Pydantic Settings (Pydantic v2) with environment variables,
.env files, and graceful fallback to dataclass when running in minimal environments.
"""
from __future__ import annotations

import os
from typing import Literal

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # Catalog & Data
        data_path: str = Field(default="data/sample_catalog.json", description="Path to hierarchical JSON catalog")

        # Hierarchical Navigation & Pruning
        max_depth: int = Field(default=4, description="Maximum tree depth to explore")
        branch_top_k: int = Field(default=2, description="Top-K child branches to preserve per parent level")
        prune_threshold: float = Field(default=0.25, description="Minimum score threshold to survive branch pruning")

        # Dense Hybrid Rescue
        enable_dense_hybrid: bool = Field(default=True, description="Enable dense embedding candidates rescue")
        dense_top_n: int = Field(default=3, description="Number of dense retrieval candidates to merge")
        hybrid_alpha: float = Field(default=0.7, description="Tree vs Dense balance factor (0.0 to 1.0)")

        # Listwise Final Selection
        final_top_k: int = Field(default=5, description="Final number of ranked results to return")

        # Decision Model
        decision_model_type: Literal["mock", "cross_encoder", "llm"] = Field(
            default="mock",
            description="Decision model engine: 'mock' (rule-based/BM25-like), 'cross_encoder', or 'llm'",
        )
        gemini_api_key: str = Field(default="", description="Optional Gemini API key for LLM-backed decision model")

        # Server
        api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
        api_port: int = Field(default=8000, description="FastAPI bind port")
        log_level: str = Field(default="INFO", description="Logging verbosity")

except ImportError:
    # Graceful fallback when pydantic-settings is not yet installed in local python environment
    from dataclasses import dataclass

    @dataclass
    class Settings:  # type: ignore[no-redef]
        data_path: str = os.getenv("DATA_PATH", "data/sample_catalog.json")
        max_depth: int = int(os.getenv("MAX_DEPTH", "4"))
        branch_top_k: int = int(os.getenv("BRANCH_TOP_K", "2"))
        prune_threshold: float = float(os.getenv("PRUNE_THRESHOLD", "0.25"))
        enable_dense_hybrid: bool = os.getenv("ENABLE_DENSE_HYBRID", "true").lower() in ("true", "1")
        dense_top_n: int = int(os.getenv("DENSE_TOP_N", "3"))
        hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.7"))
        final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
        decision_model_type: str = os.getenv("DECISION_MODEL_TYPE", "mock")
        gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        api_host: str = os.getenv("API_HOST", "0.0.0.0")
        api_port: int = int(os.getenv("API_PORT", "8000"))
        log_level: str = os.getenv("LOG_LEVEL", "INFO")


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
