"""
GraphRAG-Nexus — BM25 Keyword Store
Keyword-based retrieval using BM25 algorithm.
Complements semantic search with exact matching.
"""

import os
import pickle
import logging
from rank_bm25 import BM25Okapi
from vectorstore.faiss_store.faiss_store import DocumentChunk
from config.settings import settings

logger = logging.getLogger(__name__)


class BM25Store:
    """
    BM25 keyword search store.
    Catches exact matches that semantic search misses.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks: list[DocumentChunk] = []
        self.tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer — lowercase and split."""
        return text.lower().split()

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add chunks to BM25 index."""
        if not chunks:
            return 0

        new_tokenized = [
            self._tokenize(chunk.text) for chunk in chunks
        ]

        self.chunks.extend(chunks)
        self.tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Added {len(chunks)} chunks to BM25 index")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = None
    ) -> list[dict]:
        """Search BM25 index for relevant chunks."""
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is empty")
            return []

        top_k = top_k or settings.bm25_top_k

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "retrieval_method": "bm25",
                    "rank": rank + 1
                })

        logger.info(
            f"BM25 search returned {len(results)} results"
        )
        return results

    def save(self, path: str = None):
        """Save BM25 index to disk."""
        path = path or settings.bm25_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "tokenized_corpus": self.tokenized_corpus
            }, f)

        logger.info(f"BM25 index saved: {len(self.chunks)} chunks")

    def load(self, path: str = None) -> bool:
        """Load BM25 index from disk."""
        path = path or settings.bm25_index_path

        if not os.path.exists(f"{path}.pkl"):
            logger.warning(f"No BM25 index found at {path}")
            return False

        try:
            with open(f"{path}.pkl", "rb") as f:
                data = pickle.load(f)

            self.chunks = data["chunks"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            logger.info(
                f"BM25 index loaded: {len(self.chunks)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def get_stats(self) -> dict:
        """Return BM25 store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "corpus_size": len(self.tokenized_corpus),
            "index_built": self.bm25 is not None,
        }
