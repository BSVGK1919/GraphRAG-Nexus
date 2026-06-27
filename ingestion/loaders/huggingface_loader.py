"""
GraphRAG-Nexus — HuggingFace Loader
Fetches model cards and dataset info
from HuggingFace Hub.
"""

import logging
import requests
from ingestion.chunkers.text_chunker import RawDocument
from config.settings import settings

logger = logging.getLogger(__name__)

HF_MODELS = [
    "bert-base-uncased",
    "gpt2",
    "facebook/bart-large-cnn",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/flan-t5-base",
    "microsoft/deberta-v3-base",
    "facebook/opt-1.3b",
    "tiiuae/falcon-7b",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b-hf",
]

HF_DATASETS = [
    "squad",
    "glue",
    "imdb",
    "common_voice",
    "cnn_dailymail",
]


class HuggingFaceLoader:
    """Loads model cards from HuggingFace Hub."""

    def __init__(self):
        self.headers = {}
        if settings.hf_token:
            self.headers["Authorization"] = (
                f"Bearer {settings.hf_token}"
            )

    def load(self) -> list[RawDocument]:
        """Load HuggingFace model cards."""
        documents = []

        for model_id in HF_MODELS:
            try:
                doc = self._fetch_model_card(model_id)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(
                    f"HF model {model_id} failed: {e}"
                )

        logger.info(
            f"HuggingFace total: {len(documents)} models"
        )
        return documents

    def _fetch_model_card(
        self,
        model_id: str
    ) -> RawDocument:
        """Fetch a single model card."""
        url = (
            f"https://huggingface.co/api/models/{model_id}"
        )
        response = requests.get(
            url,
            headers=self.headers,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        text = (
            f"Model: {model_id}\n\n"
            f"Pipeline: "
            f"{data.get('pipeline_tag', 'unknown')}\n\n"
            f"Tags: "
            f"{', '.join(data.get('tags', []))}\n\n"
            f"Downloads: {data.get('downloads', 0)}\n\n"
            f"Description: {model_id} is a "
            f"{data.get('pipeline_tag', 'ML')} model "
            f"available on HuggingFace Hub."
        )

        return RawDocument(
            doc_id=f"hf_{model_id.replace('/', '_')}",
            text=text,
            source=f"https://huggingface.co/{model_id}",
            source_type="huggingface",
            metadata={
                "model_id": model_id,
                "pipeline_tag": data.get(
                    "pipeline_tag", "unknown"
                ),
                "tags": data.get("tags", []),
                "downloads": data.get("downloads", 0)
            }
        )
