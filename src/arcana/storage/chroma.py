"""ChromaDB-backed vector store for production use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.storage.base import VectorSearchResult, VectorStore

if TYPE_CHECKING:
    import chromadb


def _require_chromadb() -> None:
    """Raise a clear error if chromadb is not installed."""
    try:
        import chromadb  # noqa: F401, F811
    except ImportError:
        raise ImportError(
            "chromadb is required for ChromaVectorStore. "
            "Install it with: pip install 'arcana[chromadb]'"
        ) from None


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-backed vector store.

    Supports both ephemeral (in-memory) and persistent modes. Uses cosine
    similarity for distance calculation.

    Requires: pip install chromadb

    Examples::

        # Ephemeral (tests / development)
        store = ChromaVectorStore()

        # Persistent (production)
        store = ChromaVectorStore(persist_directory="/data/chroma")

        await store.initialize()
    """

    def __init__(
        self,
        *,
        persist_directory: str | None = None,
        collection_name: str = "arcana_vectors",
    ) -> None:
        _require_chromadb()
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None  # type: ignore[name-defined]
        self._collection: chromadb.Collection | None = None

    async def initialize(self) -> None:
        """Create ChromaDB client and collection."""
        import chromadb

        if self._persist_directory:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def close(self) -> None:
        """Release references (ChromaDB handles cleanup internally)."""
        self._collection = None
        self._client = None

    def _get_collection(self) -> chromadb.Collection:
        """Get the collection, raising if not initialized."""
        if self._collection is None:
            raise RuntimeError(
                "ChromaVectorStore not initialized. Call initialize() first."
            )
        return self._collection

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        """Insert or update a vector with optional metadata and content."""
        collection = self._get_collection()
        kwargs: dict[str, Any] = {
            "ids": [id],
            "embeddings": [embedding],
        }
        if metadata:
            # ChromaDB requires flat metadata values (str, int, float, bool)
            kwargs["metadatas"] = [_flatten_metadata(metadata)]
        if content is not None:
            kwargs["documents"] = [content]
        collection.upsert(**kwargs)

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using cosine similarity."""
        collection = self._get_collection()

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["embeddings", "metadatas", "documents", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = collection.query(**kwargs)

        # ChromaDB returns distances; for cosine space: score = 1 - distance
        output: list[VectorSearchResult] = []
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]

        for i, vec_id in enumerate(ids):
            score = 1.0 - distances[i]
            if score < min_score:
                continue
            output.append(
                VectorSearchResult(
                    id=vec_id,
                    score=score,
                    metadata=dict(metadatas[i]) if metadatas[i] else {},
                    content=documents[i] if documents else None,
                )
            )

        return output

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID. Returns True if it existed."""
        collection = self._get_collection()
        # Check existence first since ChromaDB delete is silent
        existing = collection.get(ids=[id])
        if not existing["ids"]:
            return False
        collection.delete(ids=[id])
        return True

    async def count(self) -> int:
        """Get the total number of stored vectors."""
        collection = self._get_collection()
        return collection.count()


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Flatten metadata values for ChromaDB compatibility.

    ChromaDB only supports str, int, float, bool as metadata values.
    Complex types are converted to their string representation.
    """
    flat: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat
