"""JSONL-backed prediction registry store."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.models import PredictionRecord

logger = logging.getLogger(__name__)


class RegistryStore:
    """Append-only JSONL store for prediction records.

    Each line in the registry file is one JSON-serialised PredictionRecord.
    Updates are performed by rewriting the entire file (safe for the expected
    volume of hundreds-to-low-thousands of records).
    """

    def __init__(self, registry_path: str) -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, record: PredictionRecord) -> None:
        """Append a single record to the registry file."""
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        logger.debug("Saved prediction %s", record.prediction_id)

    def load_all(self) -> list[PredictionRecord]:
        """Load every record from the registry file."""
        if not self._path.exists():
            return []

        records: list[PredictionRecord] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(PredictionRecord(**json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed record on line %d: %s", lineno, exc
                    )
        return records

    def get_by_id(self, prediction_id: str) -> PredictionRecord | None:
        """Return the record matching *prediction_id*, or ``None``."""
        for record in self.load_all():
            if record.prediction_id == prediction_id:
                return record
        return None

    def get_by_question(
        self, question_id: str, platform: str
    ) -> list[PredictionRecord]:
        """Return all records for a given question on a given platform."""
        return [
            r
            for r in self.load_all()
            if r.question_id == question_id and r.platform == platform
        ]

    def update(self, record: PredictionRecord) -> None:
        """Replace an existing record (matched by prediction_id).

        The entire file is rewritten.  If no record with the same
        prediction_id exists, the record is appended instead.
        """
        records = self.load_all()
        found = False
        for idx, existing in enumerate(records):
            if existing.prediction_id == record.prediction_id:
                records[idx] = record
                found = True
                break

        if not found:
            logger.info(
                "prediction_id %s not found for update; appending as new record",
                record.prediction_id,
            )
            records.append(record)

        self._write_all(records)
        logger.debug("Updated prediction %s", record.prediction_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_all(self, records: list[PredictionRecord]) -> None:
        """Atomically rewrite the registry file with *records*."""
        tmp_path = self._path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(rec.model_dump_json() + "\n")
        os.replace(tmp_path, self._path)
