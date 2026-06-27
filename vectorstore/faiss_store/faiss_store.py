"""
GraphRAG-Nexus — FAISS Vector Store
Local vector store for semantic search.
"""

import os
import pickle
import logging
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    source: str
    source_type: str
    metadata: dict = field(default_factory=dict)
    embedding: list = field(default_factory=list)
    graph_nodes: list = field(default_factory=list)


class FAISSStore:
    """
    Local FAISS vector store for semantic search.
    Stores document chunks with embeddings.
    """

    def __init__(self):
        self.index = None
        self.chunks: list[DocumentChunk] = []
        self.embedding_model = None
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load sentence transformer embedding model."""
        try:
            self.embedding_model = SentenceTransformer(
                settings.embedding_model,
                cache_folder="/tmp/sentence_transformers"
            )
            logger.info(
                f"Embedding model loaded: {settings.embedding_model}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _init_index(self):
        """Initialise FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(settings.faiss_dimension)
            logger.info("FAISS index initialised")
        except Exception as e:
            logger.error(f"Failed to initialise FAISS index: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=32
        )
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the FAISS index."""
        if not chunks:
            return 0

        if self.index is None:
            self._init_index()

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to FAISS index")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = None
    ) -> list[dict]:
        """
        Search FAISS index for similar chunks.

        Returns list of dicts with chunk and score.
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("FAISS index is empty")
            return []

        top_k = top_k or settings.retrieval_top_k

        query_embedding = self.embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "score": float(score),
                "retrieval_method": "faiss",
                "rank": len(results) + 1
            })

        logger.info(
            f"FAISS search returned {len(results)} results"
        )
        return results

    def save(self, path: str = None):
        """Save FAISS index and chunks to disk."""
        path = path or settings.faiss_index_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        import faiss
        faiss.write_index(self.index, f"{path}.faiss")

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(
            f"FAISS index saved: {len(self.chunks)} chunks"
        )

    def load(self, path: str = None) -> bool:
        """Load FAISS index and chunks from disk."""
        path = path or settings.faiss_index_path

        if not os.path.exists(f"{path}.faiss"):
            logger.warning(f"No FAISS index found at {path}")
            return False

        try:
            import faiss
            self.index = faiss.read_index(f"{path}.faiss")

            with open(f"{path}.pkl", "rb") as f:
                self.chunks = pickle.load(f)

            logger.info(
                f"FAISS index loaded: {len(self.chunks)} chunks"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def get_stats(self) -> dict:
        """Return FAISS store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": settings.faiss_dimension,
            "embedding_model": settings.embedding_model,
        }
