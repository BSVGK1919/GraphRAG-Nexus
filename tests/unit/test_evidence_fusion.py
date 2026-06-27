"""
Unit tests for Evidence Fusion Layer.
"""

import pytest
from vectorstore.evidence_fusion import EvidenceFusion
from vectorstore.faiss_store.faiss_store import DocumentChunk


def make_result(
    chunk_id: str,
    text: str,
    score: float,
    method: str,
    rank: int
) -> dict:
    return {
        "chunk": DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            source="test",
            source_type="test",
        ),
        "score": score,
        "retrieval_method": method,
        "rank": rank
    }


class TestEvidenceFusion:

    def setup_method(self):
        self.fusion = EvidenceFusion()

    def test_fuse_single_list(self):
        results = [
            make_result("c1", "Python ML", 0.9, "faiss", 1),
            make_result("c2", "PyTorch DL", 0.8, "faiss", 2),
        ]
        fused = self.fusion.fuse([results])
        assert len(fused) == 2
        assert fused[0]["rank"] == 1

    def test_fuse_multiple_lists(self):
        faiss_results = [
            make_result("c1", "Python ML", 0.9, "faiss", 1),
            make_result("c2", "PyTorch DL", 0.8, "faiss", 2),
        ]
        bm25_results = [
            make_result("c1", "Python ML", 5.0, "bm25", 1),
            make_result("c3", "NLP text", 4.0, "bm25", 2),
        ]
        fused = self.fusion.fuse([faiss_results, bm25_results])
        assert len(fused) <= 5
        assert len(fused) > 0

    def test_deduplicates_same_chunk(self):
        faiss_results = [
            make_result("c1", "Python ML", 0.9, "faiss", 1),
        ]
        bm25_results = [
            make_result("c1", "Python ML", 5.0, "bm25", 1),
        ]
        fused = self.fusion.fuse([faiss_results, bm25_results])
        chunk_ids = [r["chunk"].chunk_id for r in fused]
        assert chunk_ids.count("c1") == 1

    def test_multi_source_flag(self):
        faiss_results = [
            make_result("c1", "Python ML", 0.9, "faiss", 1),
        ]
        bm25_results = [
            make_result("c1", "Python ML", 5.0, "bm25", 1),
        ]
        fused = self.fusion.fuse([faiss_results, bm25_results])
        c1_result = next(
            r for r in fused if r["chunk"].chunk_id == "c1"
        )
        assert c1_result["multi_source"] is True

    def test_fuse_empty_lists(self):
        fused = self.fusion.fuse([])
        assert fused == []

    def test_top_k_limit(self):
        results = [
            make_result(f"c{i}", f"chunk {i}", 0.9 - i * 0.1,
                       "faiss", i + 1)
            for i in range(10)
        ]
        fused = self.fusion.fuse([results], top_k=3)
        assert len(fused) <= 3

    def test_confidence_score(self):
        results = [
            make_result("c1", "Python ML", 0.9, "faiss", 1),
            make_result("c2", "PyTorch DL", 0.8, "faiss", 2),
        ]
        fused = self.fusion.fuse([results])
        confidence = self.fusion.calculate_confidence(fused)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_empty(self):
        confidence = self.fusion.calculate_confidence([])
        assert confidence == 0.0

    def test_result_structure(self):
        results = [
            make_result("c1", "Python ML", 0.9, "faiss", 1),
        ]
        fused = self.fusion.fuse([results])
        for result in fused:
            assert "chunk" in result
            assert "score" in result
            assert "retrieval_methods" in result
            assert "rank" in result
            assert "multi_source" in result
