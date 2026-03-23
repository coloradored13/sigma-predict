"""Metaculus API client."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from src.config import Config
from src.models import Platform, PlatformQuestion, QuestionType, Resolution
from src.platforms.base import PlatformClient

logger = logging.getLogger(__name__)

BASE_URL = "https://www.metaculus.com/api2"


def _parse_question_type(data: dict) -> QuestionType:
    """Map Metaculus question type to our QuestionType enum."""
    possibilities = data.get("possibilities", {})
    ptype = possibilities.get("type", "")
    if ptype == "binary":
        return QuestionType.BINARY
    if ptype == "continuous":
        return QuestionType.NUMERIC
    # Group questions or anything with sub-questions
    if data.get("group") or data.get("type") == "group":
        return QuestionType.MULTIPLE_CHOICE
    return QuestionType.BINARY


def _extract_community_prediction(data: dict) -> float | None:
    """Extract the median community prediction from a question payload."""
    try:
        return float(data["community_prediction"]["full"]["q2"])
    except (KeyError, TypeError, ValueError):
        return None


def _map_question(data: dict) -> PlatformQuestion:
    """Convert a raw Metaculus API question dict to a PlatformQuestion."""
    return PlatformQuestion(
        question_id=str(data["id"]),
        platform=Platform.METACULUS,
        title=data.get("title", ""),
        description=data.get("description", ""),
        question_type=_parse_question_type(data),
        resolution_criteria=data.get("resolution_criteria", ""),
        fine_print=data.get("fine_print", ""),
        created_at=data.get("created_at", ""),
        close_date=data.get("close_time", ""),
        resolve_date=data.get("resolve_time", ""),
        community_prediction=_extract_community_prediction(data),
        url=data.get("url", f"https://www.metaculus.com/questions/{data['id']}/"),
        tags=[t.get("name", t) if isinstance(t, dict) else str(t)
              for t in data.get("tags", [])],
    )


class MetaculusClient(PlatformClient):
    """Synchronous Metaculus API client using httpx."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        api_key = self._config.metaculus_api_key
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Token {api_key}"
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    # -- PlatformClient interface ------------------------------------------

    @property
    def platform(self) -> Platform:
        return Platform.METACULUS

    def get_questions(
        self,
        limit: int = 20,
        offset: int = 0,
        tournament_id: int | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
    ) -> list[PlatformQuestion]:
        """Fetch a page of open forecast questions from Metaculus."""
        params: dict[str, str | int] = {
            "limit": limit,
            "offset": offset,
            "status": "open",
            "type": "forecast",
            "order_by": "-activity",
        }
        if tournament_id is not None:
            params["tournament_id"] = tournament_id
        if search:
            params["search"] = search
        elif tags:
            params["search"] = " ".join(tags)

        try:
            resp = self._client.get("/questions/", params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", data) if isinstance(data, dict) else data
            return [_map_question(q) for q in results]
        except httpx.HTTPStatusError as exc:
            logger.warning("Metaculus list questions HTTP %s: %s", exc.response.status_code, exc)
            return []
        except Exception as exc:
            logger.warning("Metaculus list questions failed: %s", exc)
            return []

    def get_question(self, question_id: str) -> PlatformQuestion:
        """Fetch a single question by ID."""
        try:
            resp = self._client.get(f"/questions/{question_id}/")
            resp.raise_for_status()
            return _map_question(resp.json())
        except httpx.HTTPStatusError as exc:
            logger.error("Metaculus get question %s HTTP %s: %s", question_id, exc.response.status_code, exc)
            raise ValueError(f"Question {question_id} not found on Metaculus") from exc
        except Exception as exc:
            logger.error("Metaculus get question %s failed: %s", question_id, exc)
            raise ValueError(f"Failed to fetch question {question_id} from Metaculus") from exc

    def submit_prediction(self, question_id: str, probability: float) -> bool:
        """Submit a binary prediction to Metaculus."""
        if not self._config.metaculus_api_key:
            logger.error("Cannot submit prediction: METACULUS_API_KEY not set")
            return False

        payload = {"prediction": probability}
        try:
            resp = self._client.post(
                f"/questions/{question_id}/predict/",
                json=payload,
            )
            resp.raise_for_status()
            logger.info("Submitted prediction %.4f for Metaculus question %s", probability, question_id)
            return True
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Metaculus submit prediction HTTP %s for question %s: %s",
                exc.response.status_code, question_id, exc,
            )
            return False
        except Exception as exc:
            logger.warning("Metaculus submit prediction failed for %s: %s", question_id, exc)
            return False

    def get_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution data if the question is resolved."""
        try:
            resp = self._client.get(f"/questions/{question_id}/")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Metaculus get resolution failed for %s: %s", question_id, exc)
            return None

        resolution_value = data.get("resolution")
        if resolution_value is None:
            return None

        try:
            outcome = float(resolution_value)
        except (TypeError, ValueError):
            return None

        resolved_at = data.get("resolve_time", "")
        if not resolved_at:
            resolved_at = datetime.now(timezone.utc).isoformat()

        return Resolution(
            outcome=outcome,
            resolved_at=resolved_at,
        )

    def get_community_prediction(self, question_id: str) -> float | None:
        """Fetch the current community median prediction."""
        try:
            resp = self._client.get(f"/questions/{question_id}/")
            resp.raise_for_status()
            return _extract_community_prediction(resp.json())
        except Exception as exc:
            logger.warning("Metaculus get community prediction failed for %s: %s", question_id, exc)
            return None

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> MetaculusClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
