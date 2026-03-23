"""Tests for platform client implementations.

These tests verify the mapping/parsing logic without hitting real APIs.
"""

import pytest

from src.models import Platform, QuestionType
from src.platforms.metaculus import _map_question, _parse_question_type, _extract_community_prediction
from src.platforms.polymarket import _map_market as _map_polymarket
from src.platforms.kalshi import _map_market as _map_kalshi, _map_event


class TestMetaculusMapping:
    def test_binary_question(self):
        data = {
            "id": 12345,
            "title": "Will X happen by 2027?",
            "description": "Description here",
            "resolution_criteria": "Resolves YES if...",
            "fine_print": "Edge cases...",
            "created_at": "2026-01-01T00:00:00Z",
            "close_time": "2027-01-01T00:00:00Z",
            "resolve_time": "2027-03-01T00:00:00Z",
            "possibilities": {"type": "binary"},
            "community_prediction": {"full": {"q2": 0.65}},
            "url": "https://www.metaculus.com/questions/12345/",
            "tags": [{"name": "technology"}, {"name": "ai"}],
        }
        q = _map_question(data)
        assert q.question_id == "12345"
        assert q.platform == Platform.METACULUS
        assert q.title == "Will X happen by 2027?"
        assert q.question_type == QuestionType.BINARY
        assert q.community_prediction == pytest.approx(0.65)
        assert "technology" in q.tags
        assert "ai" in q.tags

    def test_numeric_question(self):
        data = {
            "id": 99,
            "title": "What will GDP be?",
            "possibilities": {"type": "continuous"},
            "tags": [],
        }
        assert _parse_question_type(data) == QuestionType.NUMERIC

    def test_missing_community_prediction(self):
        assert _extract_community_prediction({}) is None
        assert _extract_community_prediction({"community_prediction": None}) is None
        assert _extract_community_prediction({"community_prediction": {}}) is None


class TestPolymarketMapping:
    def test_basic_market(self):
        data = {
            "condition_id": "abc123",
            "question": "Will event occur?",
            "description": "Market description",
            "market_slug": "will-event-occur",
            "tokens": [
                {"outcome": "Yes", "price": 0.72},
                {"outcome": "No", "price": 0.28},
            ],
            "end_date_iso": "2026-12-31T00:00:00Z",
            "tags": [],
        }
        q = _map_polymarket(data)
        assert q.question_id == "abc123"
        assert q.platform == Platform.POLYMARKET
        assert q.community_prediction == pytest.approx(0.72)
        assert q.question_type == QuestionType.BINARY

    def test_no_tokens(self):
        data = {"condition_id": "xyz", "question": "Test", "tokens": []}
        q = _map_polymarket(data)
        assert q.community_prediction is None


class TestKalshiMapping:
    def test_market_with_price(self):
        data = {
            "ticker": "AAPL-26MAR-100",
            "title": "Will AAPL close above $100?",
            "rules_primary": "Resolves based on...",
            "rules_secondary": "Edge cases...",
            "last_price": 65,  # 65 cents = 0.65 probability
            "open_time": "2026-01-01T00:00:00Z",
            "close_time": "2026-03-26T00:00:00Z",
            "expiration_time": "2026-03-27T00:00:00Z",
        }
        q = _map_kalshi(data)
        assert q.question_id == "AAPL-26MAR-100"
        assert q.platform == Platform.KALSHI
        assert q.community_prediction == pytest.approx(0.65)

    def test_event_mapping(self):
        data = {
            "event_ticker": "ELECTION-2026",
            "title": "2026 Midterm Elections",
            "description": "Events related to...",
            "category": "politics",
        }
        q = _map_event(data)
        assert q.question_id == "ELECTION-2026"
        assert q.platform == Platform.KALSHI
        assert "politics" in q.tags
