"""RegistryCache — lazy-loading dict cache over RegistryStore."""

from __future__ import annotations

from src.models import PredictionRecord
from src.registry.store import RegistryStore


class RegistryCache:
    """Dirty-flag dict cache over RegistryStore.

    Loads on first access and on demand after invalidation.
    """

    def __init__(self, store: RegistryStore):
        self._store = store
        self._records: dict[str, PredictionRecord] = {}
        self._dirty = True

    def load(self) -> list[PredictionRecord]:
        if self._dirty:
            self._records = {r.prediction_id: r for r in self._store.load_all()}
            self._dirty = False
        return list(self._records.values())

    def get(self, prediction_id: str) -> PredictionRecord | None:
        self.load()
        return self._records.get(prediction_id)

    def invalidate(self) -> None:
        self._dirty = True
