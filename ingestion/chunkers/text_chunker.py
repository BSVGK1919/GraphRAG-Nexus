"""
GraphRAG-Nexus — Text Chunker
Splits documents into semantic chunks
for embedding and storage.
"""

import re
import logging
from dataclasses import dataclass, field
from vectorstore.faiss_store.faiss_store import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """Represents a raw document before chunking."""
    doc_id: str
    text: str
    source: str
    source_type: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    Splits documents into overlapping chunks
    for semantic search.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        document: RawDocument
    ) -> list[DocumentChunk]:
        """Split a document into chunks."""
        if not document.text.strip():
            return []

        # Clean text
        text = self._clean_text(document.text)

        # Split into chunks
        chunks = self._split_text(text)

        # Create DocumentChunk objects
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < self.min_chunk_size:
                continue

            chunk = DocumentChunk(
                chunk_id=f"{document.doc_id}_chunk_{i}",
                text=chunk_text,
                source=document.source,
                source_type=document.source_type,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": document.doc_id
                }
            )
            doc_chunks.append(chunk)

        logger.debug(
            f"Chunked {document.doc_id}: "
            f"{len(doc_chunks)} chunks"
        )
        return doc_chunks

    def chunk_documents(
        self,
        documents: list[RawDocument]
    ) -> list[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents "
            f"into {len(all_chunks)} chunks"
        )
        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalise text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()

        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end == len(words):
                break

            start = end - self.chunk_overlap

        return chunks
