"""
GraphRAG-Nexus — ArXiv Loader
Fetches ML/AI research papers from ArXiv.
"""

import logging
import arxiv
from ingestion.chunkers.text_chunker import RawDocument

logger = logging.getLogger(__name__)

ARXIV_QUERIES = [
    "large language model RAG retrieval augmented generation",
    "knowledge graph machine learning",
    "transformer neural network NLP",
    "graph neural network",
    "machine learning engineer skills",
    "MLOps deployment production",
    "vector database embedding semantic search",
    "fine-tuning pre-trained models",
    "reinforcement learning from human feedback",
    "multimodal AI vision language model",
]


class ArxivLoader:
    """Loads research papers from ArXiv."""

    def __init__(self, max_papers_per_query: int = 5):
        self.max_papers = max_papers_per_query

    def load(self) -> list[RawDocument]:
        """Load papers from ArXiv."""
        documents = []
        seen_ids = set()

        for query in ARXIV_QUERIES:
            try:
                papers = self._fetch_papers(query)
                for paper in papers:
                    if paper.doc_id not in seen_ids:
                        documents.append(paper)
                        seen_ids.add(paper.doc_id)
                logger.info(
                    f"ArXiv query '{query[:40]}': "
                    f"{len(papers)} papers"
                )
            except Exception as e:
                logger.warning(
                    f"ArXiv query failed: {e}"
                )

        logger.info(
            f"ArXiv total: {len(documents)} papers"
        )
        return documents

    def _fetch_papers(
        self,
        query: str
    ) -> list[RawDocument]:
        """Fetch papers for a single query."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.max_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )

        documents = []
        for paper in client.results(search):
            text = (
                f"Title: {paper.title}\n\n"
                f"Authors: "
                f"{', '.join(str(a) for a in paper.authors)}\n\n"
                f"Abstract: {paper.summary}\n\n"
                f"Categories: "
                f"{', '.join(paper.categories)}"
            )

            doc = RawDocument(
                doc_id=f"arxiv_{paper.entry_id.split('/')[-1]}",
                text=text,
                source=paper.entry_id,
                source_type="arxiv",
                metadata={
                    "title": paper.title,
                    "authors": [
                        str(a) for a in paper.authors
                    ],
                    "year": paper.published.year
                    if paper.published else None,
                    "categories": paper.categories,
                    "url": paper.entry_id
                }
            )
            documents.append(doc)

        return documents
