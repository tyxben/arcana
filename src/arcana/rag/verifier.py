"""Citation verification for the RAG pipeline."""

from __future__ import annotations

import re

from arcana.contracts.rag import RAGAnswer, RetrievalResult, VerificationResult

# Score threshold below which a citation is considered weak
_WEAK_CITATION_THRESHOLD = 0.3


class CitationVerifier:
    """
    Verifies that RAG answers are properly supported by retrieved citations.

    Checks:
        1. Answer has citations at all.
        2. Cited chunk_ids exist in the retrieved results.
        3. Coverage: ratio of sentences with citations to total sentences.
        4. Unsupported claims: sentences without nearby citations.
        5. Weak citations: citations with low relevance scores.
    """

    def __init__(self, weak_threshold: float = _WEAK_CITATION_THRESHOLD) -> None:
        self.weak_threshold = weak_threshold

    def verify(
        self,
        answer: RAGAnswer,
        retrieved_results: list[RetrievalResult],
    ) -> VerificationResult:
        """
        Verify that an answer's citations are valid and sufficient.

        Args:
            answer: The RAG answer with citations.
            retrieved_results: The retrieval results that were used.

        Returns:
            VerificationResult with validity, coverage, and issues.
        """
        # Build set of valid chunk IDs from retrieved results
        valid_chunk_ids = {r.chunk_id for r in retrieved_results}

        # Check 1: Does the answer have any citations?
        if not answer.citations:
            sentences = self._split_sentences(answer.content)
            return VerificationResult(
                valid=False,
                coverage=0.0,
                unsupported_claims=sentences if sentences else [answer.content],
            )

        # Check 2: Are cited chunk_ids valid?
        cited_chunk_ids = {c.chunk_id for c in answer.citations if c.chunk_id}
        invalid_ids = cited_chunk_ids - valid_chunk_ids

        # Check 3 & 4: Coverage and unsupported claims
        sentences = self._split_sentences(answer.content)
        if not sentences:
            sentences = [answer.content] if answer.content.strip() else []

        # Build text index from citations for matching
        citation_snippets = [c.snippet.lower() for c in answer.citations if c.snippet]
        cited_sources = {c.source.lower() for c in answer.citations if c.source}

        supported_count = 0
        unsupported: list[str] = []

        for sentence in sentences:
            if self._sentence_is_supported(sentence, citation_snippets, cited_sources):
                supported_count += 1
            else:
                unsupported.append(sentence)

        coverage = supported_count / len(sentences) if sentences else 0.0

        # Check 5: Weak citations
        weak: list[str] = []
        for citation in answer.citations:
            if citation.score < self.weak_threshold:
                label = citation.chunk_id or citation.source
                weak.append(label)

        # Determine overall validity
        valid = (
            len(invalid_ids) == 0
            and coverage > 0.0
            and len(answer.citations) > 0
        )

        return VerificationResult(
            valid=valid,
            coverage=coverage,
            unsupported_claims=unsupported,
            weak_citations=weak,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using basic regex."""
        if not text.strip():
            return []
        # Split on sentence-ending punctuation followed by whitespace or end
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _sentence_is_supported(
        self,
        sentence: str,
        citation_snippets: list[str],
        cited_sources: set[str],
    ) -> bool:
        """
        Check if a sentence has support from citations.

        A sentence is considered supported if any citation snippet shares
        significant word overlap with it.
        """
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())

        # Check for word overlap with citation snippets
        for snippet in citation_snippets:
            snippet_words = set(snippet.split())
            if not snippet_words:
                continue
            overlap = sentence_words & snippet_words
            # Consider supported if at least 2 content words overlap
            # (excluding very short common words)
            content_overlap = {w for w in overlap if len(w) > 3}
            if len(content_overlap) >= 2:
                return True

        return False
