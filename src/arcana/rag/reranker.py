"""Reranking strategies for the RAG pipeline."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

from arcana.contracts.rag import RerankConfig, RetrievalResult


class Reranker(ABC):
    """Abstract base class for result rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        config: RerankConfig,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results.

        Args:
            query: Original query text.
            results: List of retrieval results to rerank.
            config: Reranking configuration.

        Returns:
            Reranked and filtered list of results.
        """
        ...


class BM25Reranker(Reranker):
    """
    Pure Python BM25 reranker.

    Combines the original vector similarity score with a BM25 text
    relevance score:  final = alpha * vector_score + (1 - alpha) * bm25_score

    BM25 parameters:
        k1: Term frequency saturation. Default 1.5.
        b: Length normalization. Default 0.75.
        alpha: Weight for vector score vs BM25. Default 0.7.
    """

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, alpha: float = 0.7
    ) -> None:
        self.k1 = k1
        self.b = b
        self.alpha = alpha

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        config: RerankConfig,
    ) -> list[RetrievalResult]:
        """Rerank results using BM25 scoring combined with vector scores."""
        if not results:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            # No query tokens; return sorted by original score
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            return self._apply_filters(sorted_results, config)

        # Compute BM25 scores
        doc_contents = [r.content for r in results]
        bm25_scores = self._compute_bm25(query_tokens, doc_contents)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 > 0:
            norm_bm25 = [s / max_bm25 for s in bm25_scores]
        else:
            norm_bm25 = [0.0] * len(bm25_scores)

        # Combine scores
        scored: list[tuple[RetrievalResult, float]] = []
        for result, bm25_norm in zip(results, norm_bm25, strict=True):
            final_score = self.alpha * result.score + (1 - self.alpha) * bm25_norm
            # Create a copy with updated score
            updated = result.model_copy(update={"score": final_score})
            scored.append((updated, final_score))

        # Sort by final score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [item[0] for item in scored]

        return self._apply_filters(reranked, config)

    def _compute_bm25(
        self, query_tokens: list[str], documents: list[str]
    ) -> list[float]:
        """Compute BM25 scores for all documents against query tokens."""
        # Tokenize all documents
        doc_token_lists = [self._tokenize(doc) for doc in documents]
        doc_lengths = [len(tokens) for tokens in doc_token_lists]
        avg_dl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        n_docs = len(documents)

        # Document frequency for each query term
        df: Counter[str] = Counter()
        for tokens in doc_token_lists:
            unique_tokens = set(tokens)
            for token in query_tokens:
                if token in unique_tokens:
                    df[token] += 1

        # Compute BM25 for each document
        scores: list[float] = []
        for i, doc_tokens in enumerate(doc_token_lists):
            tf_counts = Counter(doc_tokens)
            dl = doc_lengths[i]
            score = 0.0

            for term in query_tokens:
                tf = tf_counts.get(term, 0)
                doc_freq = df.get(term, 0)

                # IDF with smoothing
                idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

                # BM25 TF component
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                score += idf * numerator / denominator if denominator > 0 else 0.0

            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing and punctuation removal."""
        # Remove common punctuation and split on whitespace
        cleaned = text.lower()
        tokens: list[str] = []
        for word in cleaned.split():
            # Strip punctuation from edges
            word = word.strip(".,!?;:\"'()[]{}/-")
            if word:
                tokens.append(word)
        return tokens

    def _apply_filters(
        self, results: list[RetrievalResult], config: RerankConfig
    ) -> list[RetrievalResult]:
        """Apply top_n and min_score filters."""
        filtered = [r for r in results if r.score >= config.min_score]
        return filtered[: config.top_n]
