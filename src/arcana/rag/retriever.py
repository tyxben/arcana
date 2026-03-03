"""Retrieval pipeline for the RAG system."""

from __future__ import annotations

from arcana.contracts.rag import (
    Chunk,
    Citation,
    Document,
    IngestionConfig,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)
from arcana.contracts.trace import EventType, ToolCallRecord, TraceEvent
from arcana.rag.chunker import Chunker
from arcana.rag.embedder import Embedder
from arcana.storage.base import VectorStore
from arcana.trace.writer import TraceWriter
from arcana.utils.hashing import canonical_hash


class Retriever:
    """
    RAG retrieval pipeline.

    Handles document ingestion (chunk -> embed -> store) and
    query retrieval (embed -> search -> build citations).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        trace_writer: TraceWriter | None = None,
        run_id: str = "rag",
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.trace_writer = trace_writer
        self.run_id = run_id
        self._chunker = Chunker()

    async def ingest(
        self, document: Document, config: IngestionConfig | None = None
    ) -> list[Chunk]:
        """
        Ingest a document into the vector store.

        Steps:
            1. Chunk the document using the configured strategy.
            2. Embed all chunk texts.
            3. Upsert chunks with embeddings into the vector store.

        Args:
            document: Source document to ingest.
            config: Ingestion configuration (uses defaults if None).

        Returns:
            List of chunks that were created and stored.
        """
        if config is None:
            config = IngestionConfig()

        # 1. Chunk
        chunks = self._chunker.chunk(document, config)
        if not chunks:
            return []

        # 2. Embed
        texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed(texts)

        # 3. Store embeddings on chunks and upsert
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding
            chunk.embedding_model = config.embedding_model
            await self.vector_store.upsert(
                id=chunk.id,
                embedding=embedding,
                metadata={
                    "document_id": chunk.document_id,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    **chunk.metadata,
                },
                content=chunk.content,
            )

        return chunks

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query.

        Steps:
            1. Embed the query text.
            2. Search vector store for similar chunks.
            3. Build RetrievalResult list with citations.
            4. Log to trace if trace_writer is present.

        Args:
            query: Retrieval query with parameters.

        Returns:
            RetrievalResponse with results and metadata.
        """
        # 1. Embed query
        query_embeddings = await self.embedder.embed([query.query])
        query_embedding = query_embeddings[0]

        # 2. Search
        search_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            filters=query.filters,
            min_score=query.min_score,
        )

        # 3. Build results with citations
        results: list[RetrievalResult] = []
        for sr in search_results:
            doc_id = sr.metadata.get("document_id", "") if sr.metadata else ""
            source = sr.metadata.get("source", doc_id) if sr.metadata else doc_id

            citation = Citation(
                source=source or doc_id,
                chunk_id=sr.id,
                document_id=doc_id,
                snippet=sr.content[:200] if sr.content else "",
                score=sr.score,
            )
            results.append(
                RetrievalResult(
                    chunk_id=sr.id,
                    document_id=doc_id,
                    score=sr.score,
                    content=sr.content or "",
                    citation=citation,
                )
            )

        response = RetrievalResponse(
            query=query.query,
            results=results,
            total_candidates=len(search_results),
        )

        # 4. Trace logging
        if self.trace_writer:
            self._log_retrieval(query, response)

        return response

    def _log_retrieval(
        self, query: RetrievalQuery, response: RetrievalResponse
    ) -> None:
        """Log a retrieval operation to the trace."""
        if not self.trace_writer:
            return

        event = TraceEvent(
            run_id=self.run_id,
            event_type=EventType.TOOL_CALL,
            tool_call=ToolCallRecord(
                name="rag_retrieve",
                args_digest=canonical_hash({"query": query.query, "top_k": query.top_k}),
                result_digest=canonical_hash(
                    {"count": len(response.results), "query": query.query}
                ),
                side_effect="read",
            ),
            metadata={
                "query": query.query,
                "result_count": len(response.results),
                "top_score": response.results[0].score if response.results else 0.0,
            },
        )
        self.trace_writer.write(event)
