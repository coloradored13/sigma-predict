"""Polymarket CLOB API client (read-only stub)."""

from __future__ import annotations

import logging

import httpx

from src.config import Config
from src.models import Platform, PlatformQuestion, QuestionType, Resolution
from src.platforms.base import PlatformClient

logger = logging.getLogger(__name__)

BASE_URL = "https://clob.polymarket.com"


def _map_market(data: dict) -> PlatformQuestion:
    """Convert a raw Polymarket market dict to a PlatformQuestion."""
    # Polymarket markets are binary yes/no by default
    question_type = QuestionType.BINARY

    # Best-effort community prediction from token prices
    community_pred: float | None = None
    tokens = data.get("tokens", [])
    for token in tokens:
        if token.get("outcome", "").lower() == "yes":
            price = token.get("price")
            if price is not None:
                try:
                    community_pred = float(price)
                except (TypeError, ValueError):
                    pass
            break

    condition_id = data.get("condition_id", data.get("id", ""))
    market_slug = data.get("market_slug", "")
    url = f"https://polymarket.com/event/{market_slug}" if market_slug else ""

    return PlatformQuestion(
        question_id=str(condition_id),
        platform=Platform.POLYMARKET,
        title=data.get("question", data.get("title", "")),
        description=data.get("description", ""),
        question_type=question_type,
        resolution_criteria="",
        fine_print="",
        created_at=data.get("created_at", ""),
        close_date=data.get("end_date_iso", data.get("close_time", "")),
        resolve_date=data.get("end_date_iso", ""),
        community_prediction=community_pred,
        url=url,
        tags=data.get("tags", []),
    )


class PolymarketClient(PlatformClient):
    """Read-only Polymarket CLOB API client.

    Supports listing and fetching markets. Prediction submission is
    not yet implemented.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._config.polymarket_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    # -- PlatformClient interface ------------------------------------------

    @property
    def platform(self) -> Platform:
        return Platform.POLYMARKET

    def get_questions(
        self,
        limit: int = 20,
        offset: int = 0,
        tournament_id: int | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
    ) -> list[PlatformQuestion]:
        """Fetch a page of markets from Polymarket."""
        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
        }
        try:
            resp = self._client.get("/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))
            return [_map_market(m) for m in markets[:limit]]
        except httpx.HTTPStatusError as exc:
            logger.warning("Polymarket list markets HTTP %s: %s", exc.response.status_code, exc)
            return []
        except Exception as exc:
            logger.warning("Polymarket list markets failed: %s", exc)
            return []

    def get_question(self, question_id: str) -> PlatformQuestion:
        """Fetch a single market by condition ID."""
        try:
            resp = self._client.get(f"/markets/{question_id}")
            resp.raise_for_status()
            return _map_market(resp.json())
        except httpx.HTTPStatusError as exc:
            logger.error("Polymarket get market %s HTTP %s: %s", question_id, exc.response.status_code, exc)
            raise ValueError(f"Market {question_id} not found on Polymarket") from exc
        except Exception as exc:
            logger.error("Polymarket get market %s failed: %s", question_id, exc)
            raise ValueError(f"Failed to fetch market {question_id} from Polymarket") from exc

    def submit_prediction(self, question_id: str, probability: float) -> bool:
        """Not implemented for Polymarket."""
        raise NotImplementedError("Polymarket submission not yet implemented")

    def get_resolution(self, question_id: str) -> Resolution | None:
        """Resolution retrieval not yet implemented for Polymarket."""
        return None

    def get_community_prediction(self, question_id: str) -> float | None:
        """Fetch current market price as the community prediction."""
        try:
            resp = self._client.get(f"/markets/{question_id}")
            resp.raise_for_status()
            data = resp.json()
            tokens = data.get("tokens", [])
            for token in tokens:
                if token.get("outcome", "").lower() == "yes":
                    price = token.get("price")
                    if price is not None:
                        return float(price)
        except Exception as exc:
            logger.warning("Polymarket get community prediction failed for %s: %s", question_id, exc)
        return None

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
