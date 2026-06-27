"""
Unit tests for FAISS Vector Store.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vectorstore.faiss_store.faiss_store import FAISSStore, DocumentChunk


class TestFAISSStore:

    @patch("vectorstore.faiss_store.faiss_store.SentenceTransformer")
    def setup_method(self, method, mock_st):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(
            5, 384
        ).astype(np.float32)
        mock_st.return_value = mock_model
        self.store = FAISSStore()
        self.store.embedding_model = mock_model

    def _make_chunks(self, n: int) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"chunk_{i}",
                text=f"Python PyTorch machine learning chunk {i}",
                source=f"source_{i}",
                source_type="arxiv",
                metadata={"index": i}
            )
            for i in range(n)
        ]

    def test_add_chunks(self):
        chunks = self._make_chunks(5)
        self.store.embedding_model.encode.return_value = (
            np.random.rand(5, 384).astype(np.float32)
        )
        added = self.store.add_chunks(chunks)
        assert added == 5

    def test_add_empty_chunks(self):
        added = self.store.add_chunks([])
        assert added == 0

    def test_search_returns_results(self):
        chunks = self._make_chunks(5)
        self.store.embedding_model.encode.return_value = (
            np.random.rand(5, 384).astype(np.float32)
        )
        self.store.add_chunks(chunks)

        self.store.embedding_model.encode.return_value = (
            np.random.rand(1, 384).astype(np.float32)
        )
        results = self.store.search("machine learning", top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_search_empty_store(self):
        self.store.embedding_model.encode.return_value = (
            np.random.rand(1, 384).astype(np.float32)
        )
        results = self.store.search("machine learning")
        assert results == []

    def test_search_result_structure(self):
        chunks = self._make_chunks(3)
        self.store.embedding_model.encode.return_value = (
            np.random.rand(3, 384).astype(np.float32)
        )
        self.store.add_chunks(chunks)

        self.store.embedding_model.encode.return_value = (
            np.random.rand(1, 384).astype(np.float32)
        )
        results = self.store.search("python", top_k=2)

        for result in results:
            assert "chunk" in result
            assert "score" in result
            assert "retrieval_method" in result
            assert result["retrieval_method"] == "faiss"

    def test_get_stats(self):
        stats = self.store.get_stats()
        assert "total_chunks" in stats
        assert "dimension" in stats
        assert "embedding_model" in stats

    def test_save_and_load(self, tmp_path):
        chunks = self._make_chunks(3)
        self.store.embedding_model.encode.return_value = (
            np.random.rand(3, 384).astype(np.float32)
        )
        self.store.add_chunks(chunks)

        save_path = str(tmp_path / "test_index")
        self.store.save(save_path)

        new_store = FAISSStore.__new__(FAISSStore)
        new_store.index = None
        new_store.chunks = []
        new_store.embedding_model = self.store.embedding_model

        loaded = new_store.load(save_path)
        assert loaded is True
        assert len(new_store.chunks) == 3
