"""Embedding providers for the RAG pipeline."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import httpx


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        ...


class OpenAIEmbedder(Embedder):
    """
    Embedding provider using OpenAI-compatible API.

    Calls POST /v1/embeddings directly via httpx (no openai SDK).
    Works with OpenAI, Azure, or any compatible endpoint.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com",
        model_name: str = "text-embedding-ada-002",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI-compatible API."""
        if not texts:
            return []

        url = f"{self.base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model_name,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()

        data = response.json()
        # Sort by index to maintain input order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]


class MockEmbedder(Embedder):
    """
    Deterministic mock embedder for testing.

    Generates fixed-length embedding vectors from text content using
    hashlib. Same text always produces the same embedding.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic fake embeddings from text hashes."""
        return [self._hash_to_vector(text) for text in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic float vector using SHA-256.

        Uses hash bytes interpreted as unsigned integers mapped to [-1, 1].
        This avoids NaN/inf issues from struct.unpack float interpretation.
        """
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            hash_input = f"{text}:{counter}".encode()
            digest = hashlib.sha256(hash_input).digest()
            # Each byte -> float in [-1, 1]
            for byte_val in digest:
                values.append((byte_val / 127.5) - 1.0)
            counter += 1

        # Truncate to exact dimensions
        return values[: self.dimensions]
