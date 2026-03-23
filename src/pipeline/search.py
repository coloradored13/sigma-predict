"""Search module with Tavily backend for evidence gathering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from src.config import Config
from src.models import SearchPersona

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"

# Persona-specific query templates.
# Each persona maps to a list of query-generation strategies that transform
# the raw question into search queries tuned for that persona's evidence style.
_PERSONA_QUERY_STRATEGIES: dict[SearchPersona, list[str]] = {
    SearchPersona.ACADEMIC: [
        "{question} scholarly research",
        "{question} historical base rate data",
        "{question} academic study findings",
        "{question} reference class frequency",
        "{question} systematic review evidence",
    ],
    SearchPersona.NEWS: [
        "{question} latest news",
        "{question} expert commentary analysis",
        "{question} current developments",
        "{question} recent updates timeline",
        "{question} breaking news implications",
    ],
    SearchPersona.CONTRARIAN: [
        "why might {question} NOT happen",
        "{question} risks obstacles unlikely",
        "evidence against {question}",
        "{question} contrarian view skeptic",
        "{question} failure scenarios downside",
    ],
    SearchPersona.DOMAIN_EXPERT: [
        "{question} government data official statistics",
        "{question} industry report technical analysis",
        "{question} domain expert assessment",
        "{question} regulatory filing data",
        "{question} technical feasibility constraints",
    ],
    SearchPersona.META_FORECASTER: [
        "{question} prediction market prices odds",
        "{question} forecaster commentary superforecaster",
        "{question} Metaculus Polymarket probability",
        "{question} forecast accuracy track record",
        "{question} prediction platform consensus",
    ],
}


@dataclass
class SearchResult:
    """A single search result from the Tavily API."""

    title: str
    url: str
    content: str
    score: float


class SearchModule:
    """Executes web searches via the Tavily API."""

    def __init__(self, config: Config) -> None:
        self._api_key = config.tavily_api_key
        self._search_depth = config.search.search_depth
        self._default_max_results = config.search.max_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a single search query and return parsed results."""
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": self._search_depth,
            "max_results": max_results,
            "include_raw_content": False,
            "include_answer": True,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    TAVILY_API_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Tavily API HTTP error %s: %s", exc.response.status_code, exc)
            return []
        except httpx.RequestError as exc:
            logger.error("Tavily API request failed: %s", exc)
            return []

        results: list[SearchResult] = []
        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                )
            )

        return results

    def search_with_persona(
        self, question: str, persona: SearchPersona
    ) -> list[SearchResult]:
        """Generate persona-tailored queries and return deduplicated results.

        Each persona produces 3-5 search queries designed around its evidence
        strategy (academic, news, contrarian, domain expert, meta-forecaster).
        Results are deduplicated by URL so the same source is not returned
        multiple times across queries.
        """
        templates = _PERSONA_QUERY_STRATEGIES.get(persona, [])
        if not templates:
            logger.warning("No query strategies for persona %s; falling back to raw search", persona)
            return self.search(question, max_results=self._default_max_results)

        seen_urls: set[str] = set()
        deduplicated: list[SearchResult] = []

        for template in templates:
            query = template.format(question=question)
            hits = self.search(query, max_results=self._default_max_results)
            for hit in hits:
                if hit.url not in seen_urls:
                    seen_urls.add(hit.url)
                    deduplicated.append(hit)

        return deduplicated
