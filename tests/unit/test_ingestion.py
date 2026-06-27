"""
Unit tests for ingestion pipeline components.
"""

import pytest
from ingestion.chunkers.text_chunker import (
    TextChunker, RawDocument
)
from ingestion.loaders.static_loader import StaticLoader


class TestTextChunker:

    def setup_method(self):
        self.chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=10
        )

    def _make_doc(self, text: str) -> RawDocument:
        return RawDocument(
            doc_id="test_doc",
            text=text,
            source="test",
            source_type="test"
        )

    def test_chunks_short_text(self):
        doc = self._make_doc("Python is great for ML")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_chunks_long_text(self):
        text = " ".join([f"word{i}" for i in range(200)])
        doc = self._make_doc(text)
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_ids_unique(self):
        text = " ".join([f"word{i}" for i in range(200)])
        doc = self._make_doc(text)
        chunks = self.chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_preserves_source(self):
        doc = self._make_doc("Python machine learning")
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.source == "test"
            assert chunk.source_type == "test"

    def test_empty_document(self):
        doc = self._make_doc("")
        chunks = self.chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_multiple_documents(self):
        docs = [
            self._make_doc(
                " ".join([f"word{i}" for i in range(100)])
            )
            for _ in range(3)
        ]
        chunks = self.chunker.chunk_documents(docs)
        assert len(chunks) > 3

    def test_chunk_metadata(self):
        doc = self._make_doc("Python ML framework")
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "doc_id" in chunk.metadata

    def test_clean_text(self):
        text = "Python   is   great\n\n\nfor ML"
        cleaned = self.chunker._clean_text(text)
        assert "   " not in cleaned


class TestStaticLoader:

    def setup_method(self):
        self.loader = StaticLoader()

    def test_loads_documents(self):
        docs = self.loader.load()
        assert len(docs) > 0

    def test_documents_have_content(self):
        docs = self.loader.load()
        for doc in docs:
            assert len(doc.text) > 100

    def test_documents_have_source_type(self):
        docs = self.loader.load()
        for doc in docs:
            assert doc.source_type == "static"

    def test_documents_have_ids(self):
        docs = self.loader.load()
        ids = [doc.doc_id for doc in docs]
        assert len(ids) == len(set(ids))

    def test_ml_roadmap_included(self):
        docs = self.loader.load()
        texts = " ".join([doc.text for doc in docs])
        assert "Machine Learning" in texts

    def test_python_libraries_included(self):
        docs = self.loader.load()
        texts = " ".join([doc.text for doc in docs])
        assert "PyTorch" in texts
