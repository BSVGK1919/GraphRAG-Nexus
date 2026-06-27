"""
GraphRAG-Nexus — Main Ingestion Pipeline
Orchestrates full data ingestion:
1. Load from all sources
2. Upload raw to S3
3. Chunk documents
4. Extract entities + generate triples
5. Load into Neo4j
6. Generate embeddings
7. Store in FAISS + Pinecone
8. Build BM25 index
9. Save all indexes
"""

import os
import time
import logging
import boto3
from botocore.exceptions import ClientError

from config.settings import settings
from ingestion.loaders.arxiv_loader import ArxivLoader
from ingestion.loaders.huggingface_loader import HuggingFaceLoader
from ingestion.loaders.adzuna_loader import AdzunaLoader
from ingestion.loaders.static_loader import StaticLoader
from ingestion.chunkers.text_chunker import TextChunker
from ingestion.extractors.entity_extractor import EntityExtractor
from ingestion.extractors.triple_generator import TripleGenerator
from ingestion.graph_builder.neo4j_loader import Neo4jLoader
from vectorstore.faiss_store.faiss_store import FAISSStore
from vectorstore.bm25_store.bm25_store import BM25Store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Full ingestion pipeline.
    Loads, processes, and stores all knowledge.
    """

    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=64
        )
        self.extractor = EntityExtractor()
        self.triple_generator = TripleGenerator()
        self.neo4j_loader = Neo4jLoader()
        self.faiss_store = FAISSStore()
        self.bm25_store = BM25Store()

        # Load existing indexes
        self.faiss_store.load()
        self.bm25_store.load()

        # S3 client
        self.s3 = None
        self._init_s3()

    def _init_s3(self):
        """Initialise S3 client."""
        try:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            logger.info("S3 client initialised")
        except Exception as e:
            logger.warning(f"S3 init failed: {e}")
            self.s3 = None

    def run(self) -> dict:
        """Run the full ingestion pipeline."""
        start_time = time.time()
        stats = {
            "sources": {},
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_triples": 0,
            "total_neo4j_nodes": 0,
            "errors": []
        }

        logger.info("=" * 60)
        logger.info("GraphRAG-Nexus Ingestion Pipeline Starting")
        logger.info("=" * 60)

        # ── Step 1: Load from all sources ────────────────
        all_documents = []

        sources = [
            ("static", StaticLoader()),
            ("arxiv", ArxivLoader(max_papers_per_query=5)),
            ("huggingface", HuggingFaceLoader()),
            ("adzuna", AdzunaLoader(max_jobs_per_search=10)),
        ]

        for source_name, loader in sources:
            try:
                logger.info(f"[{source_name}] Loading...")
                docs = loader.load()
                all_documents.extend(docs)
                stats["sources"][source_name] = len(docs)
                logger.info(
                    f"[{source_name}] Loaded {len(docs)} docs"
                )
            except Exception as e:
                logger.error(
                    f"[{source_name}] Failed: {e}"
                )
                stats["errors"].append(
                    f"{source_name}: {str(e)}"
                )
                stats["sources"][source_name] = 0

        stats["total_documents"] = len(all_documents)
        logger.info(
            f"Total documents loaded: {len(all_documents)}"
        )

        if not all_documents:
            logger.error("No documents loaded. Exiting.")
            return stats

        # ── Step 2: Upload raw to S3 ──────────────────────
        if self.s3:
            self._upload_to_s3(all_documents)

        # ── Step 3: Chunk documents ───────────────────────
        logger.info("Chunking documents...")
        all_chunks = self.chunker.chunk_documents(all_documents)
        stats["total_chunks"] = len(all_chunks)
        logger.info(f"Total chunks: {len(all_chunks)}")

        # ── Step 4: Extract entities + triples ───────────
        logger.info("Extracting entities and triples...")
        all_entities = []
        all_relationships = []

        for doc in all_documents:
            try:
                extracted = self.extractor.extract(
                    doc.text,
                    source=doc.source_type
                )
                all_entities.extend(extracted["entities"])
                all_relationships.extend(
                    extracted["relationships"]
                )
            except Exception as e:
                logger.warning(
                    f"Entity extraction failed: {e}"
                )

        stats["total_entities"] = len(all_entities)
        logger.info(f"Total entities: {len(all_entities)}")

        # Generate RDF triples
        self.triple_generator.add_entities(all_entities)
        self.triple_generator.add_relationships(
            all_relationships
        )
        triple_stats = self.triple_generator.get_stats()
        stats["total_triples"] = triple_stats["total_triples"]

        # Save RDF triples to disk
        os.makedirs(
            settings.rdf_output_path,
            exist_ok=True
        )
        rdf_path = (
            f"{settings.rdf_output_path}/triples.ttl"
        )
        self.triple_generator.serialize(rdf_path)
        logger.info(
            f"RDF triples saved: {stats['total_triples']}"
        )

        # ── Step 5: Load into Neo4j ───────────────────────
        logger.info("Loading into Neo4j...")
        try:
            if self.neo4j_loader.test_connection():
                self.neo4j_loader.load_entities(all_entities)
                self.neo4j_loader.load_relationships(
                    all_relationships
                )
                neo4j_stats = self.neo4j_loader.get_stats()
                stats["total_neo4j_nodes"] = (
                    neo4j_stats["total_nodes"]
                )
                logger.info(
                    f"Neo4j nodes: "
                    f"{neo4j_stats['total_nodes']}"
                )
            else:
                logger.warning("Neo4j connection failed")
        except Exception as e:
            logger.error(f"Neo4j loading failed: {e}")
            stats["errors"].append(f"neo4j: {str(e)}")

        # ── Step 6+7: Embed and store in FAISS + BM25 ────
        logger.info("Generating embeddings...")
        try:
            self.faiss_store.add_chunks(all_chunks)
            self.bm25_store.add_chunks(all_chunks)
            logger.info(
                f"Chunks indexed: {len(all_chunks)}"
            )
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            stats["errors"].append(f"embedding: {str(e)}")

        # ── Step 8: Pinecone (if enabled) ────────────────
        if settings.use_pinecone:
            logger.info("Uploading to Pinecone...")
            try:
                from vectorstore.pinecone_store.pinecone_store\
                    import PineconeStore
                pinecone_store = PineconeStore()
                pinecone_store.add_chunks(all_chunks)
                logger.info("Pinecone upload complete")
            except Exception as e:
                logger.warning(f"Pinecone failed: {e}")

        # ── Step 9: Save indexes ──────────────────────────
        logger.info("Saving indexes...")
        try:
            os.makedirs(
                os.path.dirname(settings.faiss_index_path),
                exist_ok=True
            )
            self.faiss_store.save()
            self.bm25_store.save()
            logger.info("Indexes saved")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            stats["errors"].append(f"save: {str(e)}")

        # ── Done ──────────────────────────────────────────
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 1)

        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info(f"Documents:  {stats['total_documents']}")
        logger.info(f"Chunks:     {stats['total_chunks']}")
        logger.info(f"Entities:   {stats['total_entities']}")
        logger.info(f"Triples:    {stats['total_triples']}")
        logger.info(f"Neo4j nodes:{stats['total_neo4j_nodes']}")
        logger.info(f"Time:       {elapsed:.1f}s")
        if stats["errors"]:
            logger.warning(f"Errors: {stats['errors']}")
        logger.info("=" * 60)

        return stats

    def _upload_to_s3(self, documents: list):
        """Upload raw documents to S3."""
        import json
        try:
            data = [
                {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "source": doc.source,
                    "source_type": doc.source_type,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            self.s3.put_object(
                Bucket=settings.aws_s3_bucket_raw,
                Key="documents/raw_documents.json",
                Body=json.dumps(data, indent=2),
                ContentType="application/json"
            )
            logger.info(
                f"Uploaded {len(documents)} docs to S3"
            )
        except Exception as e:
            logger.warning(f"S3 upload failed: {e}")


def run_pipeline():
    """Entry point for ingestion pipeline."""
    pipeline = IngestionPipeline()
    stats = pipeline.run()
    return stats


if __name__ == "__main__":
    run_pipeline()
