"""
Unit tests for BM25 Store.
"""

import pytest
from vectorstore.bm25_store.bm25_store import BM25Store
from vectorstore.faiss_store.faiss_store import DocumentChunk


class TestBM25Store:

    def setup_method(self):
        self.store = BM25Store()

    def _make_chunks(self, texts: list[str]) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"chunk_{i}",
                text=text,
                source=f"source_{i}",
                source_type="test",
                metadata={}
            )
            for i, text in enumerate(texts)
        ]

    def test_add_chunks(self):
        chunks = self._make_chunks([
            "Python machine learning tutorial",
            "PyTorch deep learning framework",
            "TensorFlow neural networks guide"
        ])
        added = self.store.add_chunks(chunks)
        assert added == 3

    def test_add_empty_chunks(self):
        added = self.store.add_chunks([])
        assert added == 0

    def test_search_returns_relevant_results(self):
        chunks = self._make_chunks([
            "Python is a programming language for ML",
            "PyTorch is a deep learning framework",
            "Cooking recipes for pasta and pizza",
        ])
        self.store.add_chunks(chunks)
        results = self.store.search("Python machine learning")
        assert len(results) > 0
        assert results[0]["retrieval_method"] == "bm25"

    def test_search_exact_match(self):
        chunks = self._make_chunks([
            "FAISS vector similarity search",
            "Random text about cooking",
            "Neo4j graph database queries",
        ])
        self.store.add_chunks(chunks)
        results = self.store.search("FAISS vector")
        assert len(results) > 0
        assert "faiss" in results[0]["chunk"].text.lower()

    def test_search_empty_store(self):
        results = self.store.search("machine learning")
        assert results == []

    def test_search_result_structure(self):
        chunks = self._make_chunks([
            "GraphRAG knowledge graph system",
            "Vector embeddings for semantic search",
        ])
        self.store.add_chunks(chunks)
        results = self.store.search("knowledge graph")

        for result in results:
            assert "chunk" in result
            assert "score" in result
            assert "retrieval_method" in result
            assert "rank" in result

    def test_save_and_load(self, tmp_path):
        chunks = self._make_chunks([
            "Machine learning with Python",
            "Deep learning frameworks comparison",
        ])
        self.store.add_chunks(chunks)

        save_path = str(tmp_path / "bm25_index")
        self.store.save(save_path)

        new_store = BM25Store()
        loaded = new_store.load(save_path)
        assert loaded is True
        assert len(new_store.chunks) == 2

    def test_get_stats(self):
        chunks = self._make_chunks(["Test chunk one"])
        self.store.add_chunks(chunks)
        stats = self.store.get_stats()
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 1
        assert stats["index_built"] is True
