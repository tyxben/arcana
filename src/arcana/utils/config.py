"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ModelProviderConfig(BaseModel):
    """Configuration for a model provider."""

    api_key: str | None = None
    base_url: str | None = None


class TraceConfig(BaseModel):
    """Configuration for the trace system."""

    enabled: bool = True
    directory: Path = Path("./traces")


class DefaultModelConfig(BaseModel):
    """Default model configuration."""

    provider: str = "gemini"
    model_id: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_ms: int = 30000


class BudgetConfig(BaseModel):
    """Budget limits configuration."""

    max_tokens_per_run: int = 100000
    max_cost_per_run_usd: float = 1.0


class ArcanaConfig(BaseModel):
    """Main configuration for Arcana."""

    # Provider configs
    gemini: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    deepseek: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    openai: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    anthropic: ModelProviderConfig = Field(default_factory=ModelProviderConfig)

    # System configs
    trace: TraceConfig = Field(default_factory=TraceConfig)
    default_model: DefaultModelConfig = Field(default_factory=DefaultModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


def load_config(env_file: str | Path | None = None) -> ArcanaConfig:
    """
    Load configuration from environment variables.

    Args:
        env_file: Path to .env file. If None, looks for .env in current directory.

    Returns:
        ArcanaConfig instance with loaded values
    """
    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    def get_env(key: str, default: Any = None) -> Any:
        return os.environ.get(key, default)

    def get_bool(key: str, default: bool = False) -> bool:
        val = get_env(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes")

    def get_int(key: str, default: int) -> int:
        val = get_env(key)
        if val is None:
            return default
        return int(val)

    def get_float(key: str, default: float) -> float:
        val = get_env(key)
        if val is None:
            return default
        return float(val)

    return ArcanaConfig(
        gemini=ModelProviderConfig(
            api_key=get_env("GEMINI_API_KEY"),
            base_url=get_env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
        ),
        deepseek=ModelProviderConfig(
            api_key=get_env("DEEPSEEK_API_KEY"),
            base_url=get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        openai=ModelProviderConfig(
            api_key=get_env("OPENAI_API_KEY"),
        ),
        anthropic=ModelProviderConfig(
            api_key=get_env("ANTHROPIC_API_KEY"),
        ),
        trace=TraceConfig(
            enabled=get_bool("TRACE_ENABLED", True),
            directory=Path(get_env("TRACE_DIR", "./traces")),
        ),
        default_model=DefaultModelConfig(
            provider=get_env("DEFAULT_PROVIDER", "gemini"),
            model_id=get_env("DEFAULT_MODEL", "gemini-2.0-flash"),
            temperature=get_float("DEFAULT_TEMPERATURE", 0.0),
            max_tokens=get_int("DEFAULT_MAX_TOKENS", 4096),
            timeout_ms=get_int("DEFAULT_TIMEOUT_MS", 30000),
        ),
        budget=BudgetConfig(
            max_tokens_per_run=get_int("MAX_TOKENS_PER_RUN", 100000),
            max_cost_per_run_usd=get_float("MAX_COST_PER_RUN_USD", 1.0),
        ),
    )


# Singleton config instance
_config: ArcanaConfig | None = None


def get_config() -> ArcanaConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
