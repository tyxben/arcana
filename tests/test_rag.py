"""Comprehensive tests for the RAG pipeline."""

from __future__ import annotations

import math
from typing import Any

import pytest

from arcana.contracts.rag import (
    ChunkingStrategy,
    Citation,
    Document,
    IngestionConfig,
    RAGAnswer,
    RerankConfig,
    RetrievalQuery,
    RetrievalResult,
    VerificationResult,
)
from arcana.rag.chunker import Chunker
from arcana.rag.embedder import MockEmbedder
from arcana.rag.reranker import BM25Reranker
from arcana.rag.retriever import Retriever
from arcana.rag.verifier import CitationVerifier
from arcana.storage.base import VectorSearchResult, VectorStore


# ── MockVectorStore ──────────────────────────────────────────────
# Inline mock since InMemoryVectorStore may not exist yet.


class MockVectorStore(VectorStore):
    """In-memory vector store for testing."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        self._data[id] = {
            "embedding": embedding,
            "metadata": metadata or {},
            "content": content or "",
        }

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        results: list[tuple[str, float]] = []
        for doc_id, doc in self._data.items():
            score = self._cosine_similarity(query_embedding, doc["embedding"])
            if score >= min_score:
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [
            VectorSearchResult(
                id=doc_id,
                score=score,
                metadata=self._data[doc_id]["metadata"],
                content=self._data[doc_id]["content"],
            )
            for doc_id, score in results[:top_k]
        ]

    async def delete(self, id: str) -> bool:
        if id in self._data:
            del self._data[id]
            return True
        return False

    async def count(self) -> int:
        return len(self._data)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_document() -> Document:
    """A sample document for testing."""
    return Document(
        id="doc-1",
        source="test.txt",
        content=(
            "Machine learning is a subset of artificial intelligence. "
            "It allows computers to learn from data without being explicitly programmed.\n\n"
            "Deep learning is a subset of machine learning. "
            "It uses neural networks with multiple layers.\n\n"
            "Natural language processing deals with the interaction "
            "between computers and human language."
        ),
        metadata={"author": "test", "topic": "AI"},
    )


@pytest.fixture
def chunker() -> Chunker:
    return Chunker()


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dimensions=128)


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


# ── TestChunker ──────────────────────────────────────────────────


class TestChunkerFixed:
    """Test FIXED chunking strategy."""

    def test_basic_fixed_split(self, chunker: Chunker) -> None:
        doc = Document(id="d1", source="s", content="a" * 100)
        config = IngestionConfig(
            chunk_size=30, chunk_overlap=5, chunking_strategy=ChunkingStrategy.FIXED
        )
        chunks = chunker.chunk(doc, config)
        assert len(chunks) >= 3
        # All chunks should have content
        for c in chunks:
            assert len(c.content) > 0
            assert c.document_id == "d1"

    def test_fixed_overlap(self, chunker: Chunker) -> None:
        doc = Document(id="d1", source="s", content="abcdefghij" * 5)  # 50 chars
        config = IngestionConfig(
            chunk_size=20, chunk_overlap=5, chunking_strategy=ChunkingStrategy.FIXED
        )
        chunks = chunker.chunk(doc, config)
        # Verify overlap: end of chunk N overlaps with start of chunk N+1
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            tail = chunks[i].content[-5:]
            head = chunks[i + 1].content[:5]
            assert tail == head

    def test_fixed_single_chunk(self, chunker: Chunker) -> None:
        doc = Document(id="d1", source="s", content="short text")
        config = IngestionConfig(
            chunk_size=100, chunk_overlap=10, chunking_strategy=ChunkingStrategy.FIXED
        )
        chunks = chunker.chunk(doc, config)
        assert len(chunks) == 1
        assert chunks[0].content == "short text"

    def test_empty_document(self, chunker: Chunker) -> None:
        doc = Document(id="d1", source="s", content="")
        config = IngestionConfig(chunking_strategy=ChunkingStrategy.FIXED)
        chunks = chunker.chunk(doc, config)
        assert len(chunks) == 0


class TestChunkerParagraph:
    """Test PARAGRAPH chunking strategy."""

    def test_paragraph_split(self, chunker: Chunker, sample_document: Document) -> None:
        config = IngestionConfig(
            chunk_size=512, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        chunks = chunker.chunk(sample_document, config)
        assert len(chunks) == 3  # Three paragraphs

    def test_paragraph_preserves_content(
        self, chunker: Chunker, sample_document: Document
    ) -> None:
        config = IngestionConfig(
            chunk_size=512, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        chunks = chunker.chunk(sample_document, config)
        # Each chunk should be a complete paragraph
        assert "Machine learning" in chunks[0].content
        assert "Deep learning" in chunks[1].content
        assert "Natural language" in chunks[2].content

    def test_paragraph_oversized_splits(self, chunker: Chunker) -> None:
        # One huge paragraph should be sub-split
        doc = Document(id="d1", source="s", content="word " * 200)
        config = IngestionConfig(
            chunk_size=50, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        chunks = chunker.chunk(doc, config)
        assert len(chunks) > 1


class TestChunkerRecursive:
    """Test RECURSIVE chunking strategy."""

    def test_recursive_uses_paragraphs_first(
        self, chunker: Chunker, sample_document: Document
    ) -> None:
        config = IngestionConfig(
            chunk_size=150, chunking_strategy=ChunkingStrategy.RECURSIVE
        )
        chunks = chunker.chunk(sample_document, config)
        # Should get paragraph-based chunks (content has 3 paragraphs)
        assert len(chunks) >= 2

    def test_recursive_falls_to_sentence(self, chunker: Chunker) -> None:
        # Single paragraph with multiple sentences
        doc = Document(
            id="d1",
            source="s",
            content=(
                "First sentence is here. Second sentence follows. "
                "Third sentence is also present. Fourth sentence at the end."
            ),
        )
        config = IngestionConfig(
            chunk_size=60,
            chunk_overlap=0,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
        )
        chunks = chunker.chunk(doc, config)
        assert len(chunks) >= 2

    def test_recursive_falls_to_fixed(self, chunker: Chunker) -> None:
        # Single long word with no sentence or paragraph breaks
        doc = Document(id="d1", source="s", content="x" * 200)
        config = IngestionConfig(
            chunk_size=50,
            chunk_overlap=10,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
        )
        chunks = chunker.chunk(doc, config)
        assert len(chunks) >= 3


class TestChunkerMetadata:
    """Test metadata preservation in chunks."""

    def test_metadata_inherited(self, chunker: Chunker) -> None:
        doc = Document(
            id="d1",
            source="s",
            content="Hello world. Another sentence.",
            metadata={"author": "Alice", "version": 2},
        )
        config = IngestionConfig(
            chunk_size=512, chunking_strategy=ChunkingStrategy.FIXED
        )
        chunks = chunker.chunk(doc, config)
        assert len(chunks) >= 1
        assert chunks[0].metadata["author"] == "Alice"
        assert chunks[0].metadata["version"] == 2

    def test_chunk_ids_are_deterministic(self, chunker: Chunker) -> None:
        doc = Document(id="d1", source="s", content="Hello world test content here.")
        config = IngestionConfig(chunking_strategy=ChunkingStrategy.FIXED)
        chunks1 = chunker.chunk(doc, config)
        chunks2 = chunker.chunk(doc, config)
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id

    def test_chunk_offsets(self, chunker: Chunker) -> None:
        content = "abcdefghij" * 10  # 100 chars
        doc = Document(id="d1", source="s", content=content)
        config = IngestionConfig(
            chunk_size=30, chunk_overlap=0, chunking_strategy=ChunkingStrategy.FIXED
        )
        chunks = chunker.chunk(doc, config)
        for c in chunks:
            assert c.content == content[c.start_offset : c.end_offset]


# ── TestMockEmbedder ─────────────────────────────────────────────


class TestMockEmbedder:
    """Test the mock embedding provider."""

    async def test_embed_returns_correct_dimensions(
        self, mock_embedder: MockEmbedder
    ) -> None:
        embeddings = await mock_embedder.embed(["hello world"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 128

    async def test_embed_deterministic(self, mock_embedder: MockEmbedder) -> None:
        emb1 = await mock_embedder.embed(["test text"])
        emb2 = await mock_embedder.embed(["test text"])
        assert emb1[0] == emb2[0]

    async def test_different_texts_different_embeddings(
        self, mock_embedder: MockEmbedder
    ) -> None:
        embeddings = await mock_embedder.embed(["hello", "world"])
        assert len(embeddings) == 2
        assert embeddings[0] != embeddings[1]

    async def test_embed_empty_list(self, mock_embedder: MockEmbedder) -> None:
        embeddings = await mock_embedder.embed([])
        assert embeddings == []

    async def test_embed_multiple(self, mock_embedder: MockEmbedder) -> None:
        texts = ["alpha", "beta", "gamma"]
        embeddings = await mock_embedder.embed(texts)
        assert len(embeddings) == 3
        # All should have correct dimensions
        for emb in embeddings:
            assert len(emb) == 128

    async def test_custom_dimensions(self) -> None:
        embedder = MockEmbedder(dimensions=64)
        embeddings = await embedder.embed(["test"])
        assert len(embeddings[0]) == 64

    async def test_values_in_range(self, mock_embedder: MockEmbedder) -> None:
        embeddings = await mock_embedder.embed(["some text here"])
        for val in embeddings[0]:
            assert -1.1 <= val <= 1.1  # Mapped from byte values to ~[-1, 1]


# ── TestRetriever ────────────────────────────────────────────────


class TestRetrieverIngest:
    """Test document ingestion."""

    async def test_ingest_creates_chunks(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
        sample_document: Document,
    ) -> None:
        retriever = Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
        config = IngestionConfig(
            chunk_size=200, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        chunks = await retriever.ingest(sample_document, config)
        assert len(chunks) >= 2
        # Chunks should have embeddings
        for c in chunks:
            assert c.embedding is not None
            assert len(c.embedding) == 128

    async def test_ingest_stores_in_vector_store(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
        sample_document: Document,
    ) -> None:
        retriever = Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
        chunks = await retriever.ingest(sample_document)
        count = await mock_vector_store.count()
        assert count == len(chunks)

    async def test_ingest_empty_document(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
    ) -> None:
        doc = Document(id="empty", source="s", content="")
        retriever = Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
        chunks = await retriever.ingest(doc)
        assert len(chunks) == 0


class TestRetrieverSearch:
    """Test retrieval (ingest then search)."""

    async def test_retrieve_returns_results(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
        sample_document: Document,
    ) -> None:
        retriever = Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
        config = IngestionConfig(
            chunk_size=150, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        await retriever.ingest(sample_document, config)

        # Use min_score=-1.0 since mock embeddings produce arbitrary cosine values
        query = RetrievalQuery(query="machine learning", top_k=5, min_score=-1.0)
        response = await retriever.retrieve(query)

        assert response.query == "machine learning"
        assert len(response.results) > 0
        # Results should have citations
        for r in response.results:
            assert r.citation is not None
            assert r.chunk_id != ""

    async def test_retrieve_respects_top_k(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
        sample_document: Document,
    ) -> None:
        retriever = Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
        config = IngestionConfig(
            chunk_size=150, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        await retriever.ingest(sample_document, config)

        query = RetrievalQuery(query="learning", top_k=1, min_score=-1.0)
        response = await retriever.retrieve(query)
        assert len(response.results) <= 1

    async def test_retrieve_empty_store(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
    ) -> None:
        retriever = Retriever(vector_store=mock_vector_store, embedder=mock_embedder)
        query = RetrievalQuery(query="anything")
        response = await retriever.retrieve(query)
        assert len(response.results) == 0

    async def test_retrieve_with_trace(
        self,
        mock_vector_store: MockVectorStore,
        mock_embedder: MockEmbedder,
        sample_document: Document,
        tmp_path,
    ) -> None:
        from arcana.trace.writer import TraceWriter

        trace_writer = TraceWriter(trace_dir=tmp_path)
        retriever = Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            trace_writer=trace_writer,
            run_id="test-rag-run",
        )
        config = IngestionConfig(
            chunk_size=150, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        await retriever.ingest(sample_document, config)

        query = RetrievalQuery(query="deep learning", min_score=-1.0)
        await retriever.retrieve(query)

        # Verify trace file was written
        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0


# ── TestBM25Reranker ─────────────────────────────────────────────


class TestBM25Reranker:
    """Test BM25 reranking."""

    def _make_result(
        self, chunk_id: str, content: str, score: float
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            document_id="doc-1",
            score=score,
            content=content,
            citation=Citation(source="test", chunk_id=chunk_id, score=score),
        )

    def test_rerank_basic(self) -> None:
        reranker = BM25Reranker()
        results = [
            self._make_result("c1", "the cat sat on the mat", 0.5),
            self._make_result("c2", "dogs are wonderful pets", 0.8),
            self._make_result("c3", "cats love to play with yarn", 0.6),
        ]
        config = RerankConfig(top_n=3)
        reranked = reranker.rerank("cat", results, config)
        assert len(reranked) == 3
        # The cat-related results should rank higher after BM25
        cat_ids = {r.chunk_id for r in reranked[:2]}
        assert "c1" in cat_ids or "c3" in cat_ids

    def test_rerank_top_n(self) -> None:
        reranker = BM25Reranker()
        results = [
            self._make_result("c1", "alpha content here", 0.9),
            self._make_result("c2", "beta content here", 0.8),
            self._make_result("c3", "gamma content here", 0.7),
        ]
        config = RerankConfig(top_n=2)
        reranked = reranker.rerank("alpha", results, config)
        assert len(reranked) == 2

    def test_rerank_min_score(self) -> None:
        reranker = BM25Reranker()
        results = [
            self._make_result("c1", "relevant content", 0.9),
            self._make_result("c2", "irrelevant stuff", 0.1),
        ]
        config = RerankConfig(top_n=10, min_score=0.5)
        reranked = reranker.rerank("relevant content", results, config)
        for r in reranked:
            assert r.score >= 0.5

    def test_rerank_empty_results(self) -> None:
        reranker = BM25Reranker()
        config = RerankConfig()
        reranked = reranker.rerank("test", [], config)
        assert len(reranked) == 0

    def test_rerank_empty_query(self) -> None:
        reranker = BM25Reranker()
        results = [
            self._make_result("c1", "some content", 0.9),
            self._make_result("c2", "other content", 0.5),
        ]
        config = RerankConfig(top_n=5)
        reranked = reranker.rerank("", results, config)
        # Should still return results sorted by original score
        assert len(reranked) == 2

    def test_rerank_preserves_scores_order(self) -> None:
        reranker = BM25Reranker()
        results = [
            self._make_result("c1", "python programming language", 0.5),
            self._make_result("c2", "java programming language", 0.5),
            self._make_result("c3", "python is great for data science", 0.5),
        ]
        config = RerankConfig(top_n=3)
        reranked = reranker.rerank("python", results, config)
        # Python-containing results should score higher
        assert reranked[0].chunk_id in ("c1", "c3")

    def test_alpha_weighting(self) -> None:
        # With alpha=1.0, only vector score matters
        reranker = BM25Reranker(alpha=1.0)
        results = [
            self._make_result("c1", "completely irrelevant", 0.9),
            self._make_result("c2", "query exact match query", 0.1),
        ]
        config = RerankConfig(top_n=2)
        reranked = reranker.rerank("query", results, config)
        assert reranked[0].chunk_id == "c1"  # Higher vector score wins


# ── TestCitationVerifier ─────────────────────────────────────────


class TestCitationVerifier:
    """Test citation verification."""

    def _make_result(
        self, chunk_id: str, content: str, score: float = 0.8
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            document_id="doc-1",
            score=score,
            content=content,
            citation=Citation(source="test", chunk_id=chunk_id, score=score),
        )

    def test_valid_answer_with_citations(self) -> None:
        verifier = CitationVerifier()
        results = [
            self._make_result("c1", "Machine learning is powerful technology for data analysis"),
            self._make_result("c2", "Deep learning uses neural network architectures"),
        ]
        answer = RAGAnswer(
            content=(
                "Machine learning is a powerful technology for data analysis. "
                "Deep learning uses neural network architectures."
            ),
            citations=[
                Citation(
                    source="test",
                    chunk_id="c1",
                    snippet="Machine learning is powerful technology for data analysis",
                    score=0.9,
                ),
                Citation(
                    source="test",
                    chunk_id="c2",
                    snippet="Deep learning uses neural network architectures",
                    score=0.8,
                ),
            ],
            retrieved_chunks=["c1", "c2"],
        )
        result = verifier.verify(answer, results)
        assert result.valid is True
        assert result.coverage > 0.0

    def test_no_citations_invalid(self) -> None:
        verifier = CitationVerifier()
        results = [self._make_result("c1", "some content")]
        answer = RAGAnswer(
            content="This is a claim without citations.",
            citations=[],
        )
        result = verifier.verify(answer, results)
        assert result.valid is False
        assert result.coverage == 0.0
        assert len(result.unsupported_claims) > 0

    def test_weak_citations_flagged(self) -> None:
        verifier = CitationVerifier(weak_threshold=0.5)
        results = [self._make_result("c1", "some relevant content", score=0.2)]
        answer = RAGAnswer(
            content="Some relevant content is here.",
            citations=[
                Citation(
                    source="test",
                    chunk_id="c1",
                    snippet="some relevant content",
                    score=0.2,
                ),
            ],
        )
        result = verifier.verify(answer, results)
        assert len(result.weak_citations) > 0

    def test_invalid_chunk_id(self) -> None:
        verifier = CitationVerifier()
        results = [self._make_result("c1", "valid content")]
        answer = RAGAnswer(
            content="Some claim here.",
            citations=[
                Citation(
                    source="test",
                    chunk_id="nonexistent",
                    snippet="Some claim here",
                    score=0.9,
                ),
            ],
        )
        result = verifier.verify(answer, results)
        assert result.valid is False

    def test_empty_content(self) -> None:
        verifier = CitationVerifier()
        answer = RAGAnswer(content="", citations=[])
        result = verifier.verify(answer, [])
        assert result.valid is False

    def test_coverage_calculation(self) -> None:
        verifier = CitationVerifier()
        results = [
            self._make_result("c1", "machine learning is powerful and transformative"),
        ]
        answer = RAGAnswer(
            content=(
                "Machine learning is powerful and transformative. "
                "This is an unsupported claim. "
                "Another random sentence."
            ),
            citations=[
                Citation(
                    source="test",
                    chunk_id="c1",
                    snippet="machine learning is powerful and transformative",
                    score=0.9,
                ),
            ],
        )
        result = verifier.verify(answer, results)
        # Only 1 of 3 sentences should be supported
        assert 0.0 < result.coverage < 1.0


# ── TestEndToEnd ─────────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end tests for the full RAG pipeline."""

    async def test_ingest_and_retrieve(self) -> None:
        store = MockVectorStore()
        embedder = MockEmbedder(dimensions=128)
        retriever = Retriever(vector_store=store, embedder=embedder)

        doc = Document(
            id="doc-e2e",
            source="test.md",
            content=(
                "Python is a popular programming language.\n\n"
                "JavaScript runs in the browser.\n\n"
                "Rust provides memory safety without garbage collection."
            ),
        )

        config = IngestionConfig(
            chunk_size=60, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        chunks = await retriever.ingest(doc, config)
        assert len(chunks) >= 2

        # Retrieve
        query = RetrievalQuery(query="Python programming", top_k=3, min_score=-1.0)
        response = await retriever.retrieve(query)
        assert len(response.results) > 0
        assert response.query == "Python programming"

    async def test_ingest_retrieve_rerank(self) -> None:
        store = MockVectorStore()
        embedder = MockEmbedder(dimensions=128)
        retriever = Retriever(vector_store=store, embedder=embedder)
        reranker = BM25Reranker()

        doc = Document(
            id="doc-rerank",
            source="test.md",
            content=(
                "Cats are independent animals.\n\n"
                "Dogs are loyal companions.\n\n"
                "Fish live in water and require aquariums."
            ),
        )

        config = IngestionConfig(
            chunk_size=60, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        await retriever.ingest(doc, config)
        query = RetrievalQuery(query="cats", top_k=5, min_score=-1.0)
        response = await retriever.retrieve(query)

        config = RerankConfig(top_n=2)
        reranked = reranker.rerank("cats", response.results, config)
        assert len(reranked) <= 2

    async def test_full_pipeline_with_verification(self) -> None:
        store = MockVectorStore()
        embedder = MockEmbedder(dimensions=128)
        retriever = Retriever(vector_store=store, embedder=embedder)
        verifier = CitationVerifier()

        doc = Document(
            id="doc-verify",
            source="facts.txt",
            content=(
                "The Earth orbits the Sun in approximately 365 days.\n\n"
                "Water freezes at zero degrees Celsius."
            ),
        )

        config = IngestionConfig(
            chunk_size=80, chunking_strategy=ChunkingStrategy.PARAGRAPH
        )
        await retriever.ingest(doc, config)
        query = RetrievalQuery(query="Earth orbit", top_k=3, min_score=-1.0)
        response = await retriever.retrieve(query)

        # Simulate an answer with citations from results
        if response.results:
            answer = RAGAnswer(
                content="The Earth orbits the Sun in approximately 365 days.",
                citations=[response.results[0].citation],
                retrieved_chunks=[r.chunk_id for r in response.results],
            )
            result = verifier.verify(answer, response.results)
            assert isinstance(result, VerificationResult)
