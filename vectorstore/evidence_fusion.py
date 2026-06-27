"""
GraphRAG-Nexus — Evidence Fusion Layer
Merges and scores results from all retrieval paths.
Deduplicates and ranks by combined relevance score.
"""

import logging
from vectorstore.faiss_store.faiss_store import DocumentChunk
from config.settings import settings

logger = logging.getLogger(__name__)

# Weights for each retrieval method
RETRIEVAL_WEIGHTS = {
    "faiss": 0.4,
    "pinecone": 0.4,
    "bm25": 0.2,
    "graph": 0.5,
}


class EvidenceFusion:
    """
    Fuses evidence from multiple retrieval paths.
    Uses Reciprocal Rank Fusion (RRF) algorithm.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        results_list: list[list[dict]],
        top_k: int = None
    ) -> list[dict]:
        """
        Fuse multiple result lists using RRF.

        Args:
            results_list: List of result lists from
                         different retrieval methods
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        top_k = top_k or settings.retrieval_top_k

        if not results_list:
            return []

        # Flatten all results
        all_results = []
        for results in results_list:
            all_results.extend(results)

        if not all_results:
            return []

        # Score using RRF
        chunk_scores: dict[str, dict] = {}

        for results in results_list:
            method = results[0]["retrieval_method"] if results else "unknown"
            weight = RETRIEVAL_WEIGHTS.get(method, 0.3)

            for rank, result in enumerate(results):
                chunk = result["chunk"]
                chunk_id = self._get_chunk_id(chunk)

                rrf_score = weight * (1 / (self.k + rank + 1))

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "chunk": chunk,
                        "rrf_score": 0.0,
                        "methods": [],
                        "original_scores": {}
                    }

                chunk_scores[chunk_id]["rrf_score"] += rrf_score
                chunk_scores[chunk_id]["methods"].append(method)
                chunk_scores[chunk_id]["original_scores"][method] = (
                    result["score"]
                )

        # Sort by RRF score
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]

        # Format output
        fused = []
        for rank, result in enumerate(sorted_results):
            fused.append({
                "chunk": result["chunk"],
                "score": result["rrf_score"],
                "retrieval_methods": result["methods"],
                "original_scores": result["original_scores"],
                "rank": rank + 1,
                "multi_source": len(result["methods"]) > 1
            })

        logger.info(
            f"Evidence fusion: {len(all_results)} → "
            f"{len(fused)} results"
        )
        return fused

    def _get_chunk_id(self, chunk) -> str:
        """Get unique identifier for a chunk."""
        if isinstance(chunk, DocumentChunk):
            return chunk.chunk_id
        elif isinstance(chunk, dict):
            return chunk.get("chunk_id", str(hash(str(chunk))))
        return str(hash(str(chunk)))

    def calculate_confidence(
        self,
        fused_results: list[dict]
    ) -> float:
        """
        Calculate overall retrieval confidence score.
        Higher score = more reliable retrieval.
        """
        if not fused_results:
            return 0.0

        top_score = fused_results[0]["score"] if fused_results else 0
        multi_source_count = sum(
            1 for r in fused_results if r.get("multi_source")
        )
        multi_source_ratio = multi_source_count / len(fused_results)

        confidence = (top_score * 0.6) + (multi_source_ratio * 0.4)
        return min(confidence, 1.0)
