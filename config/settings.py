"""
GraphRAG-Nexus — Central configuration settings.
All values loaded from environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import Field



class Settings(BaseSettings):

    # ── App ──────────────────────────────────────────
    app_name: str = "GraphRAG-Nexus"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── API ──────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    # ── Anthropic ────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 2048

    # ── Ollama ───────────────────────────────────────
    ollama_host: str = "http://agentrag_ollama:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout: int = 180

    # ── Neo4j ────────────────────────────────────────
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"



    # ── FAISS ────────────────────────────────────────
    faiss_index_path: str = "data/embeddings/faiss_index"
    faiss_dimension: int = 384

    # ── Pinecone ─────────────────────────────────────

    pinecone_api_key: str = ""
    pinecone_index: str = "agentrag-index"
    pinecone_environment: str = "us-east-1"
    pinecone_dimension: int = 384
    pinecone_namespace: str = "graphrag-nexus-v2"
    pinecone_host: str = ""
    use_pinecone: bool = False


    # ── Embedding Model ──────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # ── BM25 ─────────────────────────────────────────
    bm25_index_path: str = "data/embeddings/bm25_index"
    bm25_top_k: int = 10

    # ── Retrieval ────────────────────────────────────
    retrieval_top_k: int = 5
    reranker_top_k: int = 3
    retrieval_confidence_threshold: float = 0.5
    graph_coverage_threshold: float = 0.7
    max_graph_hops: int = 3

    # ── LLM Router ───────────────────────────────────
    simple_query_max_words: int = 15
    router_fallback_to_claude: bool = True

    # ── Hallucination ────────────────────────────────
    reflection_score_threshold: float = 0.85
    reflection_max_loops: int = 3
    claim_verification_threshold: float = 0.95
    hallucination_rate_target: float = 0.02

    # ── Knowledge Graph ──────────────────────────────
    rdf_namespace: str = "https://graphrag-nexus.ai/ontology#"
    rdf_output_path: str = "data/processed/triples"
    graph_coverage_min: float = 0.70

    # ── AWS ──────────────────────────────────────────
    aws_region: str = "us_east_1"
    aws_s3_bucket_raw: str = "graphrag-nexus-raw"
    aws_s3_bucket_processed: str = "graphrag-nexus-processed"
    aws_s3_bucket_backups: str = "graphrag-nexus-backups"
    aws_s3_bucket_eval: str = "graphrag-nexus-eval"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # ── HuggingFace ──────────────────────────────────
    hf_token: str = ""

    # ── GitHub ───────────────────────────────────────
    github_token: str = ""

    # ── Adzuna ───────────────────────────────────────
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""

    # ── LangSmith ────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "graphrag-nexus"

    # ── MLflow ───────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5001"
    mlflow_experiment_name: str = "graphrag-nexus"

    # ── Redis ────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_ttl: int = 3600

    # ── Evaluation ───────────────────────────────────
    ragas_faithfulness_threshold: float = 0.85
    ragas_relevancy_threshold: float = 0.80
    deepeval_hallucination_threshold: float = 0.15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Single instance used across the entire project
settings = Settings()