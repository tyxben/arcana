"""Document chunking strategies for the RAG pipeline."""

from __future__ import annotations

import re

from arcana.contracts.rag import Chunk, ChunkingStrategy, Document, IngestionConfig
from arcana.utils.hashing import canonical_hash


class Chunker:
    """Splits documents into chunks using configurable strategies."""

    def chunk(self, document: Document, config: IngestionConfig) -> list[Chunk]:
        """
        Split a document into chunks based on the configured strategy.

        Args:
            document: Source document to split.
            config: Ingestion configuration with chunking parameters.

        Returns:
            List of Chunk objects with IDs, offsets, and inherited metadata.
        """
        strategy = config.chunking_strategy

        if strategy == ChunkingStrategy.FIXED:
            spans = self._fixed_split(document.content, config.chunk_size, config.chunk_overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            spans = self._paragraph_split(document.content, config.chunk_size)
        elif strategy == ChunkingStrategy.RECURSIVE:
            spans = self._recursive_split(document.content, config.chunk_size, config.chunk_overlap)
        else:
            # Fallback to fixed for unsupported strategies (e.g., SEMANTIC)
            spans = self._fixed_split(document.content, config.chunk_size, config.chunk_overlap)

        chunks: list[Chunk] = []
        for start, end in spans:
            text = document.content[start:end]
            if not text.strip():
                continue
            chunk_id = canonical_hash({"document_id": document.id, "start_offset": start})
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=text,
                    start_offset=start,
                    end_offset=end,
                    metadata={**document.metadata},
                )
            )

        return chunks

    # ── Strategy implementations ─────────────────────────────────

    def _fixed_split(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[int, int]]:
        """Split text into fixed-size character windows with overlap."""
        spans: list[tuple[int, int]] = []
        if not text:
            return spans

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            spans.append((start, end))
            if end >= len(text):
                break
            start = end - overlap

        return spans

    def _paragraph_split(
        self, text: str, max_chunk_size: int
    ) -> list[tuple[int, int]]:
        """Split on double newlines (paragraph boundaries)."""
        spans: list[tuple[int, int]] = []
        if not text:
            return spans

        # Split on double newlines, keeping track of positions
        paragraphs = re.split(r"\n\n+", text)
        offset = 0
        for para in paragraphs:
            # Find the actual position of this paragraph in the original text
            idx = text.find(para, offset)
            if idx == -1:
                continue
            start = idx
            end = start + len(para)

            # If a single paragraph exceeds max size, sub-split with fixed
            if len(para) > max_chunk_size:
                sub_spans = self._fixed_split(para, max_chunk_size, 0)
                for sub_start, sub_end in sub_spans:
                    spans.append((start + sub_start, start + sub_end))
            else:
                spans.append((start, end))

            offset = end

        return spans

    def _recursive_split(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[int, int]]:
        """
        Recursive strategy: try paragraph -> sentence -> fixed fallback.

        First tries to split on paragraph boundaries. If any paragraph is
        still too large, tries sentence boundaries. Falls back to fixed split.
        """
        if not text or len(text) <= chunk_size:
            return [(0, len(text))] if text else []

        # Level 1: Try paragraph split
        paragraphs = re.split(r"\n\n+", text)
        if len(paragraphs) > 1:
            return self._recursive_merge(text, paragraphs, chunk_size, overlap)

        # Level 2: Try sentence split
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self._recursive_merge(text, sentences, chunk_size, overlap)

        # Level 3: Fixed split fallback
        return self._fixed_split(text, chunk_size, overlap)

    def _recursive_merge(
        self,
        full_text: str,
        segments: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """
        Merge segments into chunks that fit within chunk_size.

        Accumulates segments until adding the next one would exceed the limit,
        then starts a new chunk.
        """
        spans: list[tuple[int, int]] = []
        offset = 0

        current_start: int | None = None
        current_end = 0

        for segment in segments:
            idx = full_text.find(segment, offset)
            if idx == -1:
                continue

            seg_start = idx
            seg_end = seg_start + len(segment)

            if current_start is None:
                current_start = seg_start
                current_end = seg_end
            elif seg_end - current_start <= chunk_size:
                # This segment fits in the current chunk
                current_end = seg_end
            else:
                # Flush current chunk
                spans.append((current_start, current_end))
                current_start = seg_start
                current_end = seg_end

            offset = seg_end

        # Flush last chunk
        if current_start is not None:
            spans.append((current_start, current_end))

        # Handle any individual oversized chunks by sub-splitting
        final_spans: list[tuple[int, int]] = []
        for start, end in spans:
            if end - start > chunk_size:
                sub = self._fixed_split(full_text[start:end], chunk_size, overlap)
                for ss, se in sub:
                    final_spans.append((start + ss, start + se))
            else:
                final_spans.append((start, end))

        return final_spans
