"""
Integration test: Real LLM provider calls through Arcana Runtime.

Requires API keys in .env or environment.
Run: uv run pytest tests/integration/test_providers_real.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def _has_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var))


class TestDeepSeekProvider:
    """Real DeepSeek API calls."""

    pytestmark = pytest.mark.skipif(
        not _has_key("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"
    )

    async def test_simple_run(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"deepseek": os.environ["DEEPSEEK_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="deepseek"),
        )
        result = await rt.run("What is 2 + 2? Answer with just the number.")
        assert result.success
        assert "4" in str(result.output)
        print(f"\nDeepSeek: {result.output} (tokens={result.tokens_used}, ${result.cost_usd:.4f})")

    async def test_sdk_run(self):
        import arcana

        result = await arcana.run(
            "Say 'hello' in Japanese. Answer with just the word.",
            provider="deepseek",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
        assert result.success
        print(f"\nDeepSeek SDK: {result.output}")


class TestOpenAIProvider:
    """Real OpenAI API calls."""

    pytestmark = pytest.mark.skipif(
        not _has_key("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )

    async def test_simple_run(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"openai": os.environ["OPENAI_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="openai"),
        )
        result = await rt.run("What is 3 * 7? Answer with just the number.")
        assert result.success
        assert "21" in str(result.output)
        print(f"\nOpenAI: {result.output} (tokens={result.tokens_used}, ${result.cost_usd:.4f})")


class TestGeminiProvider:
    """Real Gemini API calls — may fail due to region restrictions."""

    pytestmark = pytest.mark.skipif(
        not _has_key("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
    )

    async def test_simple_run(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"gemini": os.environ["GEMINI_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="gemini"),
        )
        try:
            result = await rt.run("What is 10 - 3? Answer with just the number.")
            assert result.success
            assert "7" in str(result.output)
            print(f"\nGemini: {result.output}")
        except Exception as e:
            if "location" in str(e).lower() or "PRECONDITION" in str(e):
                pytest.skip(f"Gemini region restriction: {e}")
            raise


class TestKimiProvider:
    """Real Kimi (Moonshot) API calls."""

    pytestmark = pytest.mark.skipif(
        not _has_key("KIMI_API_KEY"), reason="KIMI_API_KEY not set"
    )

    async def test_simple_run(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"kimi": os.environ["KIMI_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="kimi"),
        )
        result = await rt.run("What is 5 + 3? Answer with just the number.")
        assert result.success
        assert "8" in str(result.output)
        print(f"\nKimi: {result.output} (tokens={result.tokens_used}, ${result.cost_usd:.4f})")


class TestGLMProvider:
    """Real GLM (ZhipuAI) API calls."""

    pytestmark = pytest.mark.skipif(
        not _has_key("GLM_API_KEY"), reason="GLM_API_KEY not set"
    )

    async def test_simple_run(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"glm": os.environ["GLM_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="glm"),
        )
        result = await rt.run("What is 12 / 4? Answer with just the number.")
        assert result.success
        assert "3" in str(result.output)
        print(f"\nGLM: {result.output} (tokens={result.tokens_used}, ${result.cost_usd:.4f})")


class TestMiniMaxProvider:
    """Real MiniMax API calls."""

    pytestmark = pytest.mark.skipif(
        not _has_key("MINIMAX_API_KEY"), reason="MINIMAX_API_KEY not set"
    )

    async def test_simple_run(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"minimax": os.environ["MINIMAX_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="minimax"),
        )
        result = await rt.run("What is 9 - 2? Answer with just the number.")
        assert result.success
        assert "7" in str(result.output)
        print(f"\nMiniMax: {result.output} (tokens={result.tokens_used}, ${result.cost_usd:.4f})")


class TestOllamaProvider:
    """Real Ollama API calls — requires local Ollama running."""

    @staticmethod
    def _ollama_reachable() -> bool:
        """Check if Ollama is reachable at localhost:11434."""
        host = os.environ.get("OLLAMA_HOST", "")
        if not host:
            return False
        import socket
        try:
            parts = host.replace("http://", "").replace("https://", "").split(":")
            addr = parts[0] if parts else "127.0.0.1"
            port = int(parts[1]) if len(parts) > 1 else 11434
            with socket.create_connection((addr, port), timeout=2):
                return True
        except (OSError, ValueError):
            return False

    pytestmark = pytest.mark.skipif(
        not os.environ.get("OLLAMA_HOST"), reason="OLLAMA_HOST not set"
    )

    async def test_simple_run(self):
        if not self._ollama_reachable():
            pytest.skip("Ollama not reachable")

        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"ollama": ""},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="ollama"),
        )
        result = await rt.run("What is 2 + 2? Answer with just the number.")
        assert result.success
        assert "4" in str(result.output)
        print(f"\nOllama: {result.output} (tokens={result.tokens_used})")
