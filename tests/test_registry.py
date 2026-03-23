"""Tests for the prediction registry store and query layer."""

import json
import tempfile
from pathlib import Path

import pytest

from src.models import (
    PredictionRecord,
    Resolution,
    Submission,
)
from src.registry.store import RegistryStore
from src.registry.query import RegistryQuery


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a RegistryStore backed by a temporary file."""
    registry_path = str(tmp_path / "test_registry.jsonl")
    return RegistryStore(registry_path)


@pytest.fixture
def populated_registry(tmp_registry):
    """Registry with a few test records."""
    records = [
        PredictionRecord(
            prediction_id="pred-1",
            question_id="q1",
            platform="metaculus",
            question_text="Will X happen?",
            domain="geopolitics",
        ),
        PredictionRecord(
            prediction_id="pred-2",
            question_id="q2",
            platform="metaculus",
            question_text="Will Y happen?",
            domain="technology",
            resolution=Resolution(outcome=1.0, resolved_at="2026-03-01T00:00:00Z"),
            submitted=Submission(probability=0.7, submitted_at="2026-02-01", platform="metaculus"),
        ),
        PredictionRecord(
            prediction_id="pred-3",
            question_id="q3",
            platform="polymarket",
            question_text="Will Z happen?",
            domain="geopolitics",
            resolution=Resolution(outcome=0.0, resolved_at="2026-03-05T00:00:00Z"),
            submitted=Submission(probability=0.3, submitted_at="2026-02-05", platform="polymarket"),
        ),
    ]
    for r in records:
        tmp_registry.save(r)
    return tmp_registry


class TestRegistryStore:
    def test_save_and_load(self, tmp_registry):
        record = PredictionRecord(
            prediction_id="test-1",
            question_text="Test question",
        )
        tmp_registry.save(record)
        loaded = tmp_registry.load_all()
        assert len(loaded) == 1
        assert loaded[0].prediction_id == "test-1"

    def test_multiple_saves(self, tmp_registry):
        for i in range(5):
            tmp_registry.save(PredictionRecord(prediction_id=f"test-{i}"))
        loaded = tmp_registry.load_all()
        assert len(loaded) == 5

    def test_get_by_id(self, populated_registry):
        record = populated_registry.get_by_id("pred-2")
        assert record is not None
        assert record.question_text == "Will Y happen?"

    def test_get_by_id_not_found(self, populated_registry):
        assert populated_registry.get_by_id("nonexistent") is None

    def test_get_by_question(self, populated_registry):
        records = populated_registry.get_by_question("q1", "metaculus")
        assert len(records) == 1
        assert records[0].prediction_id == "pred-1"

    def test_update_existing(self, populated_registry):
        record = populated_registry.get_by_id("pred-1")
        assert record is not None
        record.domain = "updated_domain"
        populated_registry.update(record)

        updated = populated_registry.get_by_id("pred-1")
        assert updated is not None
        assert updated.domain == "updated_domain"
        # Total count should not change
        assert len(populated_registry.load_all()) == 3

    def test_update_nonexistent_appends(self, populated_registry):
        new_record = PredictionRecord(prediction_id="pred-new")
        populated_registry.update(new_record)
        assert len(populated_registry.load_all()) == 4

    def test_empty_registry(self, tmp_registry):
        assert tmp_registry.load_all() == []
        assert tmp_registry.get_by_id("anything") is None


class TestRegistryQuery:
    def test_get_resolved(self, populated_registry):
        query = RegistryQuery(populated_registry)
        resolved = query.get_resolved()
        assert len(resolved) == 2
        ids = {r.prediction_id for r in resolved}
        assert ids == {"pred-2", "pred-3"}

    def test_count_resolved(self, populated_registry):
        query = RegistryQuery(populated_registry)
        assert query.count_resolved() == 2

    def test_get_by_domain(self, populated_registry):
        query = RegistryQuery(populated_registry)
        geo = query.get_by_domain("geopolitics")
        assert len(geo) == 2
        tech = query.get_by_domain("technology")
        assert len(tech) == 1

    def test_get_calibration_pairs(self, populated_registry):
        query = RegistryQuery(populated_registry)
        pairs = query.get_calibration_pairs()
        assert len(pairs) == 2
        # pred-2: submitted 0.7, outcome 1.0
        # pred-3: submitted 0.3, outcome 0.0
        assert (0.7, 1.0) in pairs
        assert (0.3, 0.0) in pairs

    def test_get_recent_in_domain(self, populated_registry):
        query = RegistryQuery(populated_registry)
        recent = query.get_recent_in_domain("geopolitics", n=1)
        assert len(recent) == 1
