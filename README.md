# GraphRAG-Nexus v2 — Agentic GraphRAG System

> A production-grade AI/ML Career Knowledge Assistant powered by 8 LangGraph agents, hybrid retrieval, and knowledge graph reasoning.

---

## What It Does

GraphRAG-Nexus v2 takes a natural language query about AI/ML careers — papers, skills, jobs, concepts — and returns a grounded, cited answer by combining knowledge graph traversal, vector search, and keyword search through a multi-agent pipeline.

---

## Architecture

### 8 LangGraph Agents

| Agent | Role |
|---|---|
| QueryPlannerAgent | Understands intent and routes the query |
| RetrievalOrchestratorAgent | Coordinates hybrid retrieval across all stores |
| GraphReasonerAgent | Traverses Neo4j for multi-hop reasoning |
| EvidenceFusionAgent | Merges and reranks evidence using Reciprocal Rank Fusion (RRF) |
| GeneratorAgent | Synthesises the final answer using Claude |
| ClaimVerifierAgent | Fact-checks claims against retrieved evidence |
| ReflectionAgent | Scores answer quality and retries if confidence is low |
| CitationBuilderAgent | Constructs source attribution for every response |

### Knowledge Stores

| Store | Purpose |
|---|---|
| Neo4j Aura | Knowledge graph — entities, triples, multi-hop reasoning |
| Pinecone | Cloud vector store — 384-dim embeddings, namespace `graphrag-nexus-v2` |
| FAISS | Local vector store — fast in-memory search |
| BM25 | Keyword search — lexical retrieval fallback |

### Ingestion Pipeline

- **ArXiv** — AI/ML research papers
- **HuggingFace** — datasets
- **Adzuna** — job listings
- **Static** — curated knowledge documents

**Ingestion stats:** 136 documents · 544 chunks · 1,120 entities · 844 RDF triples · 181 Neo4j nodes

---

## Tech Stack

```
AI/ML          Claude API (claude-sonnet-4-6) · Ollama · sentence-transformers (all-MiniLM-L6-v2)
Orchestration  LangGraph · LangChain · LangSmith
Backend        FastAPI · Python 3.11 · Redis · Pydantic
Frontend       Streamlit
Graph/Vector   Neo4j Aura · Pinecone · FAISS · BM25 · rdflib · spaCy
Infrastructure AWS EC2 (t3.large) · Docker Compose · GitHub Actions · MLflow
Evaluation     RAGAS · DeepEval
```

---

## Project Structure

```
GraphRAG-Nexus/
├── agents/               # 8 LangGraph agents
├── retrieval/            # FAISS, Pinecone, BM25, Evidence Fusion
├── graph/                # Neo4j schema, SPARQL/Cypher engine, RDF ontology
├── ingestion/            # ArXiv, HuggingFace, Adzuna, static loaders
├── pipeline/             # LangGraph pipeline with reflection loops
├── api/                  # FastAPI endpoints
├── frontend/             # Streamlit chat UI
├── evaluation/           # RAGAS + DeepEval eval suite
├── requirements/
│   └── api.txt
├── docker-compose.yml
├── .github/workflows/    # CI/CD
└── .env.example
```

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- Neo4j Aura Free account
- Pinecone account (index: 384-dim)
- Anthropic API key
- AWS credentials (optional, for S3)

### Setup

```bash
git clone https://github.com/BSVGK1919/GraphRAG-Nexus.git
cd GraphRAG-Nexus
cp .env.example .env
# Fill in your credentials in .env
```

### Run with Docker

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Streamlit UI | http://localhost:8501 |
| MLflow | http://localhost:5001 |

### Ingest Knowledge Base

```bash
docker exec graphrag_api python -m ingestion.run
```

---

## Environment Variables

```env
ANTHROPIC_API_KEY=
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=
PINECONE_API_KEY=
PINECONE_INDEX=
PINECONE_NAMESPACE=graphrag-nexus-v2
LANGSMITH_API_KEY=
OLLAMA_BASE_URL=http://graphrag_ollama:11434
```

---

## Key Engineering Decisions

**Reciprocal Rank Fusion (RRF)** — evidence from Neo4j, Pinecone, FAISS, and BM25 is fused and reranked using RRF rather than simple score averaging, improving retrieval quality across heterogeneous sources.

**Reflection Loop** — the ReflectionAgent scores answer confidence post-generation and re-routes the query through the pipeline if quality is below threshold, enabling self-correcting responses.

**Hybrid Retrieval** — combining dense (Pinecone/FAISS), sparse (BM25), and graph-based (Neo4j Cypher/SPARQL) retrieval ensures neither purely semantic nor purely lexical queries fall through.

**GraphRAGState Dataclass** — all 8 agents share a single typed state object passed through the LangGraph pipeline, maintaining clean separation of concerns with full traceability via LangSmith.

---

## Deployment Notes

- Ubuntu 24.04 + Docker: remove `snapd` before building (`sudo apt purge snapd -y`) to avoid false "no space left on device" errors
- EC2 EBS volumes can be expanded live using `growpart` + `resize2fs` without instance restart
- Neo4j Aura Free uses the instance ID as the database name, not `neo4j`



---

## Author

**Sai Venkata Gopala Krishna Bubathula**
[GitHub](https://github.com/BSVGK1919) · [LinkedIn](https://www.linkedin.com/in/sai-venkata-gopala-krishna-bubathula-a05a26283/) · [Hugging Face](https://huggingface.co/BSVGK)
