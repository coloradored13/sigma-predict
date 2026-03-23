"""Kalshi API client (read-only stub)."""

from __future__ import annotations

import logging

import httpx

from src.config import Config
from src.models import Platform, PlatformQuestion, QuestionType, Resolution
from src.platforms.base import PlatformClient

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def _map_market(data: dict) -> PlatformQuestion:
    """Convert a raw Kalshi market dict to a PlatformQuestion."""
    # Kalshi markets are binary yes/no
    question_type = QuestionType.BINARY

    # Community prediction from last/yes price (0-100 cents -> 0.0-1.0)
    community_pred: float | None = None
    yes_price = data.get("last_price") or data.get("yes_ask") or data.get("yes_bid")
    if yes_price is not None:
        try:
            val = float(yes_price)
            # Kalshi prices are in cents (0-100)
            community_pred = val / 100.0 if val > 1.0 else val
        except (TypeError, ValueError):
            pass

    ticker = data.get("ticker", "")
    url = f"https://kalshi.com/markets/{ticker}" if ticker else ""

    return PlatformQuestion(
        question_id=str(data.get("ticker", data.get("id", ""))),
        platform=Platform.KALSHI,
        title=data.get("title", data.get("subtitle", "")),
        description=data.get("rules_primary", data.get("description", "")),
        question_type=question_type,
        resolution_criteria=data.get("rules_primary", ""),
        fine_print=data.get("rules_secondary", ""),
        created_at=data.get("open_time", ""),
        close_date=data.get("close_time", ""),
        resolve_date=data.get("expiration_time", ""),
        community_prediction=community_pred,
        url=url,
        tags=data.get("tags", []) if isinstance(data.get("tags"), list) else [],
    )


def _map_event(data: dict) -> PlatformQuestion:
    """Convert a raw Kalshi event dict to a PlatformQuestion.

    Events are top-level containers; we map them as questions for
    browsing purposes. Individual markets within events provide
    the actual tradeable contracts.
    """
    return PlatformQuestion(
        question_id=str(data.get("event_ticker", data.get("id", ""))),
        platform=Platform.KALSHI,
        title=data.get("title", ""),
        description=data.get("description", data.get("subtitle", "")),
        question_type=QuestionType.BINARY,
        resolution_criteria="",
        fine_print="",
        created_at="",
        close_date="",
        resolve_date="",
        community_prediction=None,
        url="",
        tags=data.get("category", "").split(",") if data.get("category") else [],
    )


class KalshiClient(PlatformClient):
    """Read-only Kalshi API client.

    Supports listing events and markets. Prediction submission is
    not yet implemented.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._config.kalshi_api_key
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
        return Platform.KALSHI

    def get_questions(
        self,
        limit: int = 20,
        offset: int = 0,
        tournament_id: int | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
    ) -> list[PlatformQuestion]:
        """Fetch markets from Kalshi.

        Uses the /events endpoint to list available events, then maps
        each event to a PlatformQuestion.
        """
        params: dict[str, str | int] = {
            "limit": limit,
            "cursor": str(offset) if offset else "",
        }
        if not params["cursor"]:
            del params["cursor"]

        try:
            resp = self._client.get("/events", params=params)
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", data) if isinstance(data, dict) else data
            if not isinstance(events, list):
                events = []
            return [_map_event(e) for e in events[:limit]]
        except httpx.HTTPStatusError as exc:
            logger.warning("Kalshi list events HTTP %s: %s", exc.response.status_code, exc)
            return []
        except Exception as exc:
            logger.warning("Kalshi list events failed: %s", exc)
            return []

    def get_question(self, question_id: str) -> PlatformQuestion:
        """Fetch a single market by ticker.

        Tries the /markets endpoint with the given ticker.
        """
        try:
            resp = self._client.get(f"/markets/{question_id}")
            resp.raise_for_status()
            data = resp.json()
            market = data.get("market", data) if isinstance(data, dict) else data
            return _map_market(market)
        except httpx.HTTPStatusError as exc:
            logger.error("Kalshi get market %s HTTP %s: %s", question_id, exc.response.status_code, exc)
            raise ValueError(f"Market {question_id} not found on Kalshi") from exc
        except Exception as exc:
            logger.error("Kalshi get market %s failed: %s", question_id, exc)
            raise ValueError(f"Failed to fetch market {question_id} from Kalshi") from exc

    def submit_prediction(self, question_id: str, probability: float) -> bool:
        """Not implemented for Kalshi."""
        raise NotImplementedError("Kalshi submission not yet implemented")

    def get_resolution(self, question_id: str) -> Resolution | None:
        """Resolution retrieval not yet implemented for Kalshi."""
        return None

    def get_community_prediction(self, question_id: str) -> float | None:
        """Fetch current market yes-price as the community prediction."""
        try:
            resp = self._client.get(f"/markets/{question_id}")
            resp.raise_for_status()
            data = resp.json()
            market = data.get("market", data) if isinstance(data, dict) else data
            yes_price = market.get("last_price") or market.get("yes_ask") or market.get("yes_bid")
            if yes_price is not None:
                val = float(yes_price)
                return val / 100.0 if val > 1.0 else val
        except Exception as exc:
            logger.warning("Kalshi get community prediction failed for %s: %s", question_id, exc)
        return None

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
