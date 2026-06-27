"""
GraphRAG-Nexus — Agent 2: Retrieval Orchestrator
Runs FAISS, Pinecone, and BM25 retrieval in parallel.
"""

import time
import logging
import concurrent.futures
from graph.state import GraphRAGState
from vectorstore.faiss_store.faiss_store import FAISSStore
from vectorstore.bm25_store.bm25_store import BM25Store
from config.settings import settings

logger = logging.getLogger(__name__)

# Singleton stores — loaded once
_faiss_store = None
_bm25_store = None


def get_faiss_store() -> FAISSStore:
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSStore()
        _faiss_store.load()
    return _faiss_store


def get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store()
        _bm25_store.load()
    return _bm25_store


class RetrievalOrchestratorAgent:
    """
    Agent 2 — Retrieval Orchestrator

    Responsibilities:
    - Run FAISS semantic search
    - Run BM25 keyword search
    - Run Pinecone cloud search (if enabled)
    - All three run in parallel
    - Store results in state
    """

    def __init__(self):
        self.faiss_store = get_faiss_store()
        self.bm25_store = get_bm25_store()

        # Pinecone optional
        self.pinecone_store = None
        if settings.use_pinecone:
            try:
                from vectorstore.pinecone_store.pinecone_store\
                    import PineconeStore
                self.pinecone_store = PineconeStore()
            except Exception as e:
                logger.warning(f"Pinecone unavailable: {e}")

    def run(self, state: GraphRAGState) -> GraphRAGState:
        """Run parallel retrieval across all stores."""
        start_time = time.time()
        logger.info(
            f"[Retrieval] Retrieving for: "
            f"{state.question[:60]}"
        )

        top_k = settings.retrieval_top_k

        # Run all retrievals in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=3
        ) as executor:

            faiss_future = executor.submit(
                self._faiss_search,
                state.question,
                top_k
            )
            bm25_future = executor.submit(
                self._bm25_search,
                state.question,
                top_k
            )
            pinecone_future = executor.submit(
                self._pinecone_search,
                state.question,
                top_k
            )

            state.faiss_results = faiss_future.result()
            state.bm25_results = bm25_future.result()
            state.pinecone_results = pinecone_future.result()

        # Calculate retrieval confidence
        total_results = (
            len(state.faiss_results) +
            len(state.bm25_results) +
            len(state.pinecone_results)
        )
        state.retrieval_confidence = min(
            total_results / (top_k * 3), 1.0
        )

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"[Retrieval] FAISS={len(state.faiss_results)} "
            f"BM25={len(state.bm25_results)} "
            f"Pinecone={len(state.pinecone_results)} "
            f"({latency:.0f}ms)"
        )
        return state

    def _faiss_search(
        self,
        query: str,
        top_k: int
    ) -> list[dict]:
        """Run FAISS semantic search."""
        try:
            return self.faiss_store.search(query, top_k)
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return []

    def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> list[dict]:
        """Run BM25 keyword search."""
        try:
            return self.bm25_store.search(query, top_k)
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def _pinecone_search(
        self,
        query: str,
        top_k: int
    ) -> list[dict]:
        """Run Pinecone cloud search."""
        if not self.pinecone_store:
            return []
        try:
            return self.pinecone_store.search(query, top_k)
        except Exception as e:
            logger.warning(f"Pinecone search failed: {e}")
            return []
