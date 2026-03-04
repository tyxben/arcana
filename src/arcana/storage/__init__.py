"""Storage abstraction layer for Arcana."""

from arcana.storage.base import StorageBackend, VectorSearchResult, VectorStore
from arcana.storage.memory import InMemoryBackend, InMemoryVectorStore

__all__ = [
    "InMemoryBackend",
    "InMemoryVectorStore",
    "StorageBackend",
    "VectorSearchResult",
    "VectorStore",
]


def get_chroma_store() -> type:
    """Lazy import to avoid requiring chromadb at import time."""
    from arcana.storage.chroma import ChromaVectorStore

    return ChromaVectorStore
