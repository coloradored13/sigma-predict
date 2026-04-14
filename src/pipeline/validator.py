"""Inter-stage pipeline validation gates.

Non-blocking by default: gates append to pipeline_warnings rather than
raising exceptions. Set Config.strict_validation=True to raise PipelineError
on gate failure (useful for testing).

Gates:
  base_rate_gate     — base_rate in [0.01, 0.99]
  probability_gate   — calibrated_probability in [0.01, 0.99]
  run_count_gate     — actual run count matches requested
  parse_failure_gate — any run has parse_failed=True
  search_quality_gate — avg search_quality_score below threshold
"""

from __future__ import annotations

import logging

from src.config import Config
from src.models import PredictionRecord

logger = logging.getLogger(__name__)

_BASE_RATE_MIN = 0.01
_BASE_RATE_MAX = 0.99
_PROB_MIN = 0.01
_PROB_MAX = 0.99
_SEARCH_QUALITY_WARN_THRESHOLD = 3  # avg score below this triggers warning


class PipelineError(Exception):
    """Raised by validator gates when strict_validation=True."""


class PipelineValidator:
    """Checks pipeline outputs at each stage and appends warnings.

    All gate methods return a list of warning strings (empty = pass).
    In strict mode they also raise PipelineError on the first failure.
    """

    def __init__(self, config: Config):
        self.config = config

    def validate_base_rate(self, record: PredictionRecord) -> list[str]:
        """Gate: base_rate must be in [0.01, 0.99]."""
        warnings: list[str] = []
        br = record.decomposition.base_rate
        if not (_BASE_RATE_MIN <= br <= _BASE_RATE_MAX):
            msg = (
                f"base_rate_gate: base_rate={br:.4f} outside valid range "
                f"[{_BASE_RATE_MIN}, {_BASE_RATE_MAX}]"
            )
            warnings.append(msg)
            logger.warning(msg)
        warnings.extend(self.validate_base_rate_instances(record))
        return self._apply(warnings)

    def validate_base_rate_instances(self, record: PredictionRecord) -> list[str]:
        """Gate: reference class instance count must meet reliability thresholds.

        Base rates from N<10 have ±30pp 95% CI — effectively uninformative.
        N<20 is soft-warn territory per reference class selection guidance.
        """
        warnings: list[str] = []
        count = record.decomposition.reference_class_instance_count
        if count is None:
            return warnings
        if count < 10:
            msg = (
                f"reference_class_instances_gate: reference class has {count} instance(s) "
                f"— base rate unreliable (±30pp 95% CI at N<10)"
            )
            warnings.append(msg)
            logger.warning(msg)
        elif count < 20:
            msg = (
                f"reference_class_instances_gate: reference class has {count} instance(s) "
                f"— base rate reliability low (recommend ≥20 instances)"
            )
            warnings.append(msg)
            logger.warning(msg)
        return warnings

    def validate_runs(self, record: PredictionRecord, requested_runs: int = 0) -> list[str]:
        """Gate: run count matches requested; parse failures flagged."""
        warnings: list[str] = []

        if requested_runs > 0 and len(record.runs) != requested_runs:
            msg = (
                f"run_count_gate: completed {len(record.runs)} run(s), "
                f"requested {requested_runs}"
            )
            warnings.append(msg)
            logger.warning(msg)

        failed_runs = [r for r in record.runs if r.parse_failed]
        if failed_runs:
            msg = (
                f"parse_failure_gate: {len(failed_runs)} of {len(record.runs)} "
                f"run(s) had parse_failed=True — probability fell back to base_rate"
            )
            warnings.append(msg)
            logger.warning(msg)

        if record.runs:
            avg_sq = sum(r.search_quality_score for r in record.runs) / len(record.runs)
            if avg_sq < _SEARCH_QUALITY_WARN_THRESHOLD:
                msg = (
                    f"search_quality_gate: avg search_quality_score={avg_sq:.1f} "
                    f"below threshold {_SEARCH_QUALITY_WARN_THRESHOLD}"
                )
                warnings.append(msg)
                logger.warning(msg)

        return self._apply(warnings)

    def validate_aggregation(self, record: PredictionRecord) -> list[str]:
        """Gate: calibrated_probability must be in [0.01, 0.99]."""
        warnings: list[str] = []
        prob = record.calibration.calibrated_probability
        if not (_PROB_MIN <= prob <= _PROB_MAX):
            msg = (
                f"probability_gate: calibrated_probability={prob:.4f} outside "
                f"valid range [{_PROB_MIN}, {_PROB_MAX}]"
            )
            warnings.append(msg)
            logger.warning(msg)
        return self._apply(warnings)

    def validate_all(
        self,
        record: PredictionRecord,
        requested_runs: int = 0,
    ) -> list[str]:
        """Run all gates and return combined warning list."""
        warnings: list[str] = []
        warnings.extend(self.validate_base_rate(record))
        warnings.extend(self.validate_runs(record, requested_runs))
        warnings.extend(self.validate_aggregation(record))
        return warnings

    def _apply(self, warnings: list[str]) -> list[str]:
        """In strict mode, raise on first warning. Always return warnings."""
        if warnings and self.config.strict_validation:
            raise PipelineError(warnings[0])
        return warnings
