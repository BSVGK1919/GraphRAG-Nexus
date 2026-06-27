"""
GraphRAG-Nexus — Static Knowledge Loader
Loads curated static knowledge documents
about AI/ML careers and skills.
"""

import logging
from ingestion.chunkers.text_chunker import RawDocument

logger = logging.getLogger(__name__)

STATIC_DOCUMENTS = [
    {
        "id": "ml_roadmap_2024",
        "title": "Machine Learning Engineer Roadmap 2024",
        "content": """
Machine Learning Engineer Roadmap 2024

Core Skills Required:
Advanced Python: decorators, generators, async programming,
context managers, metaclasses.
Statistics and Probability: probability distributions,
hypothesis testing, Bayesian statistics, statistical inference.
Linear Algebra: matrix operations, eigenvalues, SVD,
principal component analysis.
Calculus and Optimization: gradient descent, Adam optimizer,
RMSprop, learning rate scheduling.

Machine Learning Algorithms:
Supervised Learning: linear regression, logistic regression,
decision trees, random forests, gradient boosting (XGBoost,
LightGBM, CatBoost), support vector machines.
Unsupervised Learning: k-means clustering, DBSCAN,
hierarchical clustering, autoencoders, variational autoencoders.
Deep Learning: neural networks, CNNs, RNNs, LSTMs, GRUs,
transformers, attention mechanisms, BERT, GPT architectures.

Specialisations:
Natural Language Processing (NLP): text classification,
named entity recognition, question answering, text generation,
sentiment analysis, machine translation.
Computer Vision: image classification, object detection
(YOLO, Faster R-CNN), semantic segmentation, image generation.
Reinforcement Learning: Q-learning, policy gradients,
PPO, RLHF (Reinforcement Learning from Human Feedback).

Production and MLOps:
Model Deployment: FastAPI, Flask, TorchServe, TensorFlow Serving.
Containerisation: Docker, Kubernetes, container orchestration.
Cloud Platforms: AWS SageMaker, GCP Vertex AI, Azure ML.
Experiment Tracking: MLflow, Weights and Biases, DVC.
Feature Stores: Feast, Tecton, Hopsworks.
Monitoring: data drift detection, model performance monitoring.

Generative AI and LLMs:
Large Language Models: GPT-4, Claude, Llama, Mistral, Falcon.
RAG Systems: FAISS, Pinecone, Weaviate, ChromaDB.
Agent Frameworks: LangChain, LangGraph, AutoGen, CrewAI.
Fine-tuning: LoRA, QLoRA, PEFT, instruction tuning.
Prompt Engineering: chain-of-thought, few-shot learning.

Salary Ranges UK 2024:
Junior ML Engineer: £35,000 - £55,000
Mid-level ML Engineer: £55,000 - £80,000
Senior ML Engineer: £80,000 - £115,000
Principal ML Engineer: £115,000 - £150,000+
Head of AI/ML: £130,000 - £200,000+
        """
    },
    {
        "id": "graphrag_architecture",
        "title": "GraphRAG Architecture Guide",
        "content": """
GraphRAG Architecture Guide

What is GraphRAG:
GraphRAG combines knowledge graphs with RAG systems to enable
multi-hop reasoning over structured knowledge. Unlike naive RAG
which retrieves flat text chunks, GraphRAG traverses entity
relationships to answer complex queries.

Key Components:
Knowledge Graph: Neo4j stores entities and relationships as
nodes and edges. Entities include skills, roles, companies,
tools, frameworks, and domains.
RDF Triples: Resource Description Framework triples store
knowledge as subject-predicate-object statements enabling
SPARQL queries over structured data.
Vector Store: FAISS and Pinecone store dense vector embeddings
for semantic similarity search alongside the knowledge graph.
Hybrid Retrieval: Combines semantic search (FAISS/Pinecone),
keyword search (BM25), and graph traversal (Neo4j/SPARQL)
for comprehensive retrieval.

Advantages Over Naive RAG:
Multi-hop Reasoning: Can traverse relationship chains to
answer complex queries requiring multiple reasoning steps.
Entity Awareness: Understands relationships between entities
rather than treating text as isolated chunks.
Reduced Hallucination: Graph-grounded answers are traceable
to specific facts in the knowledge graph.
Explainability: Answers can be traced through graph paths
showing exactly how the conclusion was reached.

Technical Stack:
Graph Database: Neo4j 5.x with APOC plugins.
RDF Framework: RDFLib for Python triple management.
Query Language: Cypher for Neo4j, SPARQL for RDF.
Vector Search: FAISS (local) + Pinecone (cloud).
Orchestration: LangGraph for multi-agent pipeline.
        """
    },
    {
        "id": "python_ml_libraries",
        "title": "Python ML Libraries Complete Guide",
        "content": """
Python Machine Learning Libraries Guide

Core Scientific Computing:
NumPy: fundamental package for numerical computing.
Arrays, matrices, mathematical functions. Essential for ML.
Pandas: data manipulation and analysis. DataFrames,
Series, CSV/Excel handling, data cleaning.
SciPy: scientific computing. Optimization, statistics,
signal processing, linear algebra.
Matplotlib: 2D plotting and visualization. Line plots,
bar charts, scatter plots, heatmaps.
Seaborn: statistical data visualization built on Matplotlib.

Machine Learning:
Scikit-learn: most comprehensive ML library. Classification,
regression, clustering, preprocessing, model selection.
XGBoost: gradient boosting framework. Fast and accurate
for tabular data competitions.
LightGBM: gradient boosting by Microsoft. Faster than XGBoost
for large datasets.
CatBoost: gradient boosting by Yandex. Handles categorical
features automatically.

Deep Learning:
PyTorch: dynamic computation graphs. Preferred for research
and increasingly for production. Meta/Facebook.
TensorFlow: static and dynamic computation. Google.
Production deployment with TensorFlow Serving.
Keras: high-level API for TensorFlow. Easy to use,
good for prototyping.
JAX: NumPy-based with GPU support and automatic differentiation.

NLP:
Transformers (HuggingFace): pre-trained models (BERT, GPT,
T5), fine-tuning, tokenizers, datasets.
spaCy: industrial-strength NLP. NER, POS tagging,
dependency parsing, text classification.
NLTK: Natural Language Toolkit. 50+ corpora, tokenization,
stemming, classic NLP algorithms.
Gensim: topic modelling, Word2Vec, Doc2Vec.

MLOps:
MLflow: experiment tracking and model registry.
Weights and Biases: experiment tracking with rich visualizations.
DVC: data version control. Git for ML datasets and models.
Airflow: workflow orchestration for data pipelines.

Vector and Graph:
FAISS: Facebook AI Similarity Search. Fast nearest neighbour.
Pinecone: managed vector database with real-time updates.
LangChain: LLM application framework. Chains, agents, tools.
LangGraph: multi-agent orchestration with state graphs.
Neo4j: graph database for knowledge graphs.
RDFLib: RDF triple store and SPARQL queries in Python.
        """
    }
]


class StaticLoader:
    """Loads curated static knowledge documents."""

    def load(self) -> list[RawDocument]:
        """Load all static documents."""
        documents = []

        for doc_data in STATIC_DOCUMENTS:
            doc = RawDocument(
                doc_id=f"static_{doc_data['id']}",
                text=doc_data["content"].strip(),
                source=f"static/{doc_data['id']}",
                source_type="static",
                metadata={
                    "title": doc_data["title"],
                    "type": "curated"
                }
            )
            documents.append(doc)
            logger.info(
                f"Static doc loaded: {doc_data['title']}"
            )

        logger.info(
            f"Static total: {len(documents)} documents"
        )
        return documents
