"""
GraphRAG-Nexus — Adzuna Jobs Loader
Fetches real ML/AI job listings from Adzuna UK.
"""

import logging
import requests
from ingestion.chunkers.text_chunker import RawDocument
from config.settings import settings

logger = logging.getLogger(__name__)

JOB_SEARCHES = [
    "machine learning engineer",
    "data scientist",
    "AI engineer",
    "NLP engineer",
    "MLOps engineer",
    "deep learning engineer",
    "computer vision engineer",
    "LLM engineer",
]


class AdzunaLoader:
    """Loads job listings from Adzuna UK API."""

    BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search/1"

    def __init__(self, max_jobs_per_search: int = 10):
        self.max_jobs = max_jobs_per_search
        self.app_id = settings.adzuna_app_id
        self.app_key = settings.adzuna_app_key

    def load(self) -> list[RawDocument]:
        """Load job listings from Adzuna."""
        documents = []
        seen_ids = set()

        for search_term in JOB_SEARCHES:
            try:
                jobs = self._fetch_jobs(search_term)
                for job in jobs:
                    if job.doc_id not in seen_ids:
                        documents.append(job)
                        seen_ids.add(job.doc_id)
                logger.info(
                    f"Adzuna '{search_term}': "
                    f"{len(jobs)} jobs"
                )
            except Exception as e:
                logger.warning(
                    f"Adzuna search failed: {e}"
                )

        logger.info(
            f"Adzuna total: {len(documents)} jobs"
        )
        return documents

    def _fetch_jobs(
        self,
        search_term: str
    ) -> list[RawDocument]:
        """Fetch jobs for a search term."""
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": search_term,
            "where": "london",
            "results_per_page": self.max_jobs,
            "content-type": "application/json"
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=15
        )

        if response.status_code != 200:
            logger.warning(
                f"Adzuna API error: {response.status_code}"
            )
            return []

        data = response.json()
        documents = []

        for job in data.get("results", []):
            salary_min = job.get("salary_min", 0)
            salary_max = job.get("salary_max", 0)
            salary_text = ""
            if salary_min and salary_max:
                salary_text = (
                    f"Salary: £{salary_min:,.0f} - "
                    f"£{salary_max:,.0f}\n\n"
                )

            text = (
                f"Job Title: {job.get('title', '')}\n\n"
                f"Company: "
                f"{job.get('company', {}).get('display_name', '')}\n\n"
                f"Location: "
                f"{job.get('location', {}).get('display_name', '')}\n\n"
                f"{salary_text}"
                f"Description: {job.get('description', '')}"
            )

            doc = RawDocument(
                doc_id=f"adzuna_{job.get('id', '')}",
                text=text,
                source=job.get("redirect_url", ""),
                source_type="jobs",
                metadata={
                    "title": job.get("title", ""),
                    "company": job.get(
                        "company", {}
                    ).get("display_name", ""),
                    "location": job.get(
                        "location", {}
                    ).get("display_name", ""),
                    "salary_min": salary_min,
                    "salary_max": salary_max,
                    "search_term": search_term
                }
            )
            documents.append(doc)

        return documents
