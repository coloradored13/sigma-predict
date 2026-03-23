"""Registry query utilities for context injection into forecasting runs."""

from __future__ import annotations

from src.models import PredictionRecord
from src.registry.store import RegistryStore


class RegistryQuery:
    """Read-only query layer over the prediction registry.

    Provides filtered views of historical predictions for calibration,
    domain context, and performance analysis.
    """

    def __init__(self, store: RegistryStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_resolved(self) -> list[PredictionRecord]:
        """Return all records that have a resolution."""
        return [
            r for r in self._store.load_all() if r.resolution is not None
        ]

    def get_by_domain(self, domain: str) -> list[PredictionRecord]:
        """Return all records matching *domain* (case-insensitive)."""
        domain_lower = domain.lower()
        return [
            r for r in self._store.load_all() if r.domain.lower() == domain_lower
        ]

    def get_recent_in_domain(
        self, domain: str, n: int = 10
    ) -> list[PredictionRecord]:
        """Return the *n* most recent records in *domain* (by created_at)."""
        domain_records = self.get_by_domain(domain)
        # Sort descending by ISO-8601 created_at string (lexicographic ordering
        # works correctly for ISO timestamps with timezone).
        domain_records.sort(key=lambda r: r.created_at, reverse=True)
        return domain_records[:n]

    def get_calibration_pairs(self) -> list[tuple[float, float]]:
        """Return (predicted_probability, actual_outcome) pairs.

        Only includes resolved binary predictions that have a non-None
        outcome and a submitted probability.
        """
        pairs: list[tuple[float, float]] = []
        for record in self.get_resolved():
            if record.resolution is None or record.resolution.outcome is None:
                continue
            # Use the submitted probability if available, otherwise
            # fall back to the calibrated probability.
            if record.submitted is not None:
                predicted = record.submitted.probability
            else:
                predicted = record.calibration.calibrated_probability
            pairs.append((predicted, float(record.resolution.outcome)))
        return pairs

    def count_resolved(self) -> int:
        """Return the number of resolved predictions."""
        return len(self.get_resolved())
