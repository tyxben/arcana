"""PostgreSQL storage backends for production use."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from arcana.storage.base import StorageBackend, VectorSearchResult, VectorStore

if TYPE_CHECKING:
    import asyncpg


def _require_asyncpg() -> None:
    """Raise a clear error if asyncpg is not installed."""
    try:
        import asyncpg  # noqa: F811
        del asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required for PostgresBackend. "
            "Install it with: pip install asyncpg"
        ) from None


class PostgresBackend(StorageBackend):
    """
    PostgreSQL-backed storage for production deployments.

    Uses asyncpg for async connection pooling. Stores trace events,
    checkpoints, and key-value pairs in dedicated tables.

    Requires: pip install asyncpg
    """

    def __init__(self, dsn: str, *, min_pool_size: int = 2, max_pool_size: int = 10) -> None:
        _require_asyncpg()
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create connection pool and ensure tables exist."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )

        async with self._pool.acquire() as conn:
            await conn.execute(self._create_tables_sql())

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool, raising if not initialized."""
        if self._pool is None:
            raise RuntimeError("PostgresBackend not initialized. Call initialize() first.")
        return self._pool

    @staticmethod
    def _create_tables_sql() -> str:
        """SQL to create storage tables."""
        return """
        CREATE TABLE IF NOT EXISTS trace_events (
            id          BIGSERIAL PRIMARY KEY,
            run_id      TEXT NOT NULL,
            event_type  TEXT,
            event_data  JSONB NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_trace_events_run_id
            ON trace_events (run_id);
        CREATE INDEX IF NOT EXISTS idx_trace_events_type
            ON trace_events (run_id, event_type);

        CREATE TABLE IF NOT EXISTS checkpoints (
            id          BIGSERIAL PRIMARY KEY,
            run_id      TEXT NOT NULL,
            step_id     TEXT NOT NULL,
            state       JSONB NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id
            ON checkpoints (run_id);

        CREATE TABLE IF NOT EXISTS kv_store (
            namespace   TEXT NOT NULL,
            key         TEXT NOT NULL,
            value       JSONB NOT NULL,
            updated_at  TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (namespace, key)
        );
        """

    # ── Trace Events ─────────────────────────────────────────

    async def store_trace_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Store a trace event in PostgreSQL."""
        pool = self._get_pool()
        event_type = event.get("event_type")
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trace_events (run_id, event_type, event_data)
                VALUES ($1, $2, $3)
                """,
                run_id,
                event_type,
                json.dumps(event),
            )

    async def get_trace_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get trace events from PostgreSQL."""
        pool = self._get_pool()
        query = "SELECT event_data FROM trace_events WHERE run_id = $1"
        params: list[Any] = [run_id]
        param_idx = 2

        if event_type is not None:
            query += f" AND event_type = ${param_idx}"
            params.append(event_type)
            param_idx += 1

        query += " ORDER BY id ASC"

        if limit is not None:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [json.loads(row["event_data"]) for row in rows]

    # ── Checkpoints ──────────────────────────────────────────

    async def store_checkpoint(
        self, run_id: str, step_id: str, state: dict[str, Any]
    ) -> None:
        """Store a checkpoint in PostgreSQL."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints (run_id, step_id, state)
                VALUES ($1, $2, $3)
                """,
                run_id,
                step_id,
                json.dumps(state),
            )

    async def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Get the latest checkpoint for a run from PostgreSQL."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT step_id, state FROM checkpoints
                WHERE run_id = $1
                ORDER BY id DESC
                LIMIT 1
                """,
                run_id,
            )

        if row is None:
            return None

        state = json.loads(row["state"])
        return {"step_id": row["step_id"], **state}

    # ── Key-Value ────────────────────────────────────────────

    async def put(self, namespace: str, key: str, value: Any) -> None:
        """Store a key-value pair in PostgreSQL (upsert)."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO kv_store (namespace, key, value, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (namespace, key)
                DO UPDATE SET value = $3, updated_at = NOW()
                """,
                namespace,
                key,
                json.dumps(value),
            )

    async def get(self, namespace: str, key: str) -> Any | None:
        """Get a value from PostgreSQL."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM kv_store WHERE namespace = $1 AND key = $2",
                namespace,
                key,
            )

        if row is None:
            return None
        return json.loads(row["value"])

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a key-value pair from PostgreSQL."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM kv_store WHERE namespace = $1 AND key = $2",
                namespace,
                key,
            )
        # asyncpg returns "DELETE N" where N is the number of deleted rows
        return result == "DELETE 1"


class PgVectorStore(VectorStore):
    """
    pgvector-backed vector store for production deployments.

    Uses the pgvector extension for efficient similarity search.

    Requires: pip install asyncpg
    PostgreSQL must have the pgvector extension installed:
        CREATE EXTENSION IF NOT EXISTS vector;
    """

    def __init__(
        self,
        dsn: str,
        *,
        dimension: int = 1536,
        table_name: str = "vectors",
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        _require_asyncpg()
        self._dsn = dsn
        self._dimension = dimension
        self._table_name = table_name
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create connection pool and ensure the vector table exists."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )

        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create the vector table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id          TEXT PRIMARY KEY,
                    embedding   vector({self._dimension}),
                    metadata    JSONB DEFAULT '{{}}'::jsonb,
                    content     TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create an IVFFlat index for approximate nearest neighbor search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
                ON {self._table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool, raising if not initialized."""
        if self._pool is None:
            raise RuntimeError("PgVectorStore not initialized. Call initialize() first.")
        return self._pool

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        """Insert or update a vector in PostgreSQL."""
        pool = self._get_pool()
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        metadata_json = json.dumps(metadata or {})

        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table_name} (id, embedding, metadata, content)
                VALUES ($1, $2::vector, $3::jsonb, $4)
                ON CONFLICT (id)
                DO UPDATE SET
                    embedding = $2::vector,
                    metadata = $3::jsonb,
                    content = $4
                """,
                id,
                embedding_str,
                metadata_json,
                content,
            )

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using pgvector cosine distance."""
        pool = self._get_pool()
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build query with cosine similarity: 1 - cosine_distance
        query = f"""
            SELECT
                id,
                1 - (embedding <=> $1::vector) AS score,
                metadata,
                content
            FROM {self._table_name}
            WHERE 1 - (embedding <=> $1::vector) >= $2
        """
        params: list[Any] = [embedding_str, min_score]
        param_idx = 3

        # Apply metadata filters using JSONB containment
        if filters:
            for key, value in filters.items():
                query += f" AND metadata->>'{key}' = ${param_idx}"
                params.append(str(value))
                param_idx += 1

        query += f" ORDER BY embedding <=> $1::vector ASC LIMIT ${param_idx}"
        params.append(top_k)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            VectorSearchResult(
                id=row["id"],
                score=float(row["score"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                content=row["content"],
            )
            for row in rows
        ]

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._table_name} WHERE id = $1",
                id,
            )
        return result == "DELETE 1"

    async def count(self) -> int:
        """Get the total number of stored vectors."""
        pool = self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT COUNT(*) AS cnt FROM {self._table_name}")
        return int(row["cnt"]) if row else 0
