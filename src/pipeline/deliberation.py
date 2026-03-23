"""Conditional deliberation module (Phase 3).

Triggered when stdev > 0.15 or max-min spread > 0.25 across runs.
A supervisor agent reviews disagreements and conducts targeted search.

This is a Phase 3 feature — currently a stub.
"""

from __future__ import annotations

import logging

from src.config import Config
from src.models import Aggregation, RunResult

logger = logging.getLogger(__name__)


def should_deliberate(aggregation: Aggregation, config: Config) -> bool:
    """Check if deliberation should be triggered."""
    return aggregation.stdev > config.aggregation.deliberation_stdev_threshold


def deliberate(
    runs: list[RunResult],
    aggregation: Aggregation,
    config: Config,
) -> tuple[list[RunResult], Aggregation]:
    """Run conditional deliberation on high-disagreement predictions.

    Phase 3 implementation. Currently returns inputs unchanged.
    """
    logger.info(
        "Deliberation triggered (stdev=%.3f) but not yet implemented (Phase 3)",
        aggregation.stdev,
    )
    return runs, aggregation
