"""Registry query utilities for context injection into forecasting runs."""

from __future__ import annotations

from src.models import PredictionRecord
from src.registry.store import RegistryStore


def compute_brier_decomposition(
    pairs: list[tuple[float, float]], n_bins: int = 10
) -> dict | None:
    """Murphy (1973) Brier score decomposition into reliability, resolution, uncertainty.

    Args:
        pairs: List of (predicted_probability, outcome) tuples.
        n_bins: Number of bins for grouping predictions (adaptive).

    Returns:
        Dict with keys reliability, resolution, uncertainty, brier_check; or None
        if there are too few data points.
    """
    N = len(pairs)
    if N < 5:
        return None

    # Adaptive binning: at most N//3 bins, at least 2
    n_bins = min(n_bins, N // 3)
    if n_bins < 2:
        return None

    predictions = [p for p, _ in pairs]
    outcomes = [o for _, o in pairs]

    bar_o = sum(outcomes) / N

    # Assign each prediction to a bin by value
    bin_width = 1.0 / n_bins
    bins: dict[int, list] = {k: {"preds": [], "outs": []} for k in range(n_bins)}
    for pred, out in zip(predictions, outcomes):
        k = min(int(pred / bin_width), n_bins - 1)
        bins[k]["preds"].append(pred)
        bins[k]["outs"].append(out)

    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        n_k = len(bins[k]["preds"])
        if n_k == 0:
            continue
        bar_p_k = sum(bins[k]["preds"]) / n_k
        bar_o_k = sum(bins[k]["outs"]) / n_k
        reliability += n_k * (bar_p_k - bar_o_k) ** 2
        resolution += n_k * (bar_o_k - bar_o) ** 2

    reliability /= N
    resolution /= N
    uncertainty = bar_o * (1 - bar_o)

    return {
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "brier_check": round(reliability - resolution + uncertainty, 6),
    }


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

    def get_brier_decomposition_by_domain(
        self, records: list[PredictionRecord] | None = None
    ) -> dict[str, dict | None]:
        """Brier decomposition grouped by domain.

        Args:
            records: Records to analyse; defaults to all resolved records.

        Returns:
            Dict mapping domain -> decomposition dict (or None if too few points).
        """
        if records is None:
            records = self.get_resolved()

        by_domain: dict[str, list[tuple[float, float]]] = {}
        for record in records:
            if record.resolution is None or record.resolution.outcome is None:
                continue
            predicted = (
                record.submitted.probability
                if record.submitted is not None
                else record.calibration.calibrated_probability
            )
            domain = record.domain or "unknown"
            by_domain.setdefault(domain, []).append(
                (predicted, float(record.resolution.outcome))
            )

        return {
            domain: compute_brier_decomposition(pairs)
            for domain, pairs in by_domain.items()
        }

    def get_brier_decomposition_by_horizon(
        self, records: list[PredictionRecord] | None = None
    ) -> dict[str, dict | None]:
        """Brier decomposition grouped by prediction horizon.

        Buckets: '<30d', '30-180d', '>180d'.

        Args:
            records: Records to analyse; defaults to all resolved records.

        Returns:
            Dict mapping horizon bucket -> decomposition dict (or None if too few).
        """
        if records is None:
            records = self.get_resolved()

        by_horizon: dict[str, list[tuple[float, float]]] = {
            "<30d": [],
            "30-180d": [],
            ">180d": [],
        }
        for record in records:
            if record.resolution is None or record.resolution.outcome is None:
                continue
            predicted = (
                record.submitted.probability
                if record.submitted is not None
                else record.calibration.calibrated_probability
            )
            days = record.horizon_days
            if days is None or days < 30:
                bucket = "<30d"
            elif days <= 180:
                bucket = "30-180d"
            else:
                bucket = ">180d"
            by_horizon[bucket].append((predicted, float(record.resolution.outcome)))

        return {
            bucket: compute_brier_decomposition(pairs)
            for bucket, pairs in by_horizon.items()
        }
