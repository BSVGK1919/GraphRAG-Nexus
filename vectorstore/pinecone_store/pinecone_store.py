"""
GraphRAG-Nexus — Pinecone Vector Store
Cloud vector store for scalable semantic search.
"""

import logging
from vectorstore.faiss_store.faiss_store import DocumentChunk
from config.settings import settings

logger = logging.getLogger(__name__)


class PineconeStore:
    """
    Cloud Pinecone vector store.
    Used alongside FAISS for scalable search.
    """

    def __init__(self):
        self.index = None
        self.embedding_model = None
        self._init_pinecone()

    def _init_pinecone(self):
        """Initialise Pinecone connection."""
        try:
            from pinecone import Pinecone
            from sentence_transformers import SentenceTransformer

            pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index = pc.Index(settings.pinecone_index)
            self.embedding_model = SentenceTransformer(
                settings.embedding_model,
                cache_folder="/tmp/sentence_transformers"
            )
            logger.info(
                f"Pinecone connected: {settings.pinecone_index}"
            )
        except Exception as e:
            logger.error(f"Pinecone init failed: {e}")
            self.index = None

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add chunks to Pinecone index."""
        if not self.index or not chunks:
            return 0

        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32
            )

            vectors = []
            for i, chunk in enumerate(chunks):
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embeddings[i].tolist(),
                    "metadata": {
                        "text": chunk.text[:1000],
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        **chunk.metadata
                    }
                })

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=settings.pinecone_namespace
                )

            logger.info(
                f"Added {len(chunks)} chunks to Pinecone"
            )
            return len(chunks)

        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = None
    ) -> list[dict]:
        """Search Pinecone for similar chunks."""
        if not self.index:
            return []

        top_k = top_k or settings.retrieval_top_k

        try:
            query_embedding = self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )

            results = self.index.query(
                vector=query_embedding[0].tolist(),
                top_k=top_k,
                namespace=settings.pinecone_namespace,
                include_metadata=True
            )

            formatted = []
            for match in results.matches:
                formatted.append({
                    "chunk": DocumentChunk(
                        chunk_id=match.id,
                        text=match.metadata.get("text", ""),
                        source=match.metadata.get("source", ""),
                        source_type=match.metadata.get(
                            "source_type", ""
                        ),
                        metadata=match.metadata
                    ),
                    "score": float(match.score),
                    "retrieval_method": "pinecone",
                    "rank": len(formatted) + 1
                })

            logger.info(
                f"Pinecone search returned {len(formatted)} results"
            )
            return formatted

        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Return Pinecone index statistics."""
        if not self.index:
            return {"status": "not connected"}
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespace": settings.pinecone_namespace,
                "dimension": settings.pinecone_dimension,
            }
        except Exception as e:
            return {"error": str(e)}
