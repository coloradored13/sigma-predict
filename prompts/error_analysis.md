# Cross-Examination Error Analysis — Phase 2

> **Status**: Placeholder for Phase 2 implementation.

This prompt will drive post-resolution error analysis, examining predictions after their outcomes are known to extract structured lessons and classify error types.

## Planned Functionality

The error analysis prompt will receive:
- The full prediction record including all runs, aggregation, and calibration data
- The resolved outcome (0 or 1 for binary questions)
- The Brier score and log score for this prediction

The error analyst will:
1. Classify the error type from a taxonomy (base rate miss, evidence weighting error, actor model failure, black swan, timing error, etc.)
2. Identify which step in the pipeline contributed most to the error
3. Determine if the error was foreseeable given available evidence at prediction time
4. Extract structured lessons that can improve future predictions
5. Flag systematic patterns if this error type has occurred before

## Output Schema (Draft)

```json
{
  "error_classes": ["base_rate_miss", "evidence_weighting"],
  "primary_error_source": "decomposition | base_rate | evidence_gathering | forecaster | aggregation | calibration",
  "was_foreseeable": true,
  "foreseeable_reasoning": "What signals were available but underweighted",
  "structured_lessons": [
    "Specific, actionable lesson for future predictions"
  ],
  "suggested_pipeline_adjustments": [
    "Concrete suggestion for improving the pipeline"
  ],
  "pattern_match": "Description of similar past errors, if any"
}
```

## Implementation Notes

- Error taxonomy should be developed iteratively as predictions resolve.
- This module feeds into the calibration system — systematic error patterns should trigger recalibration.
- Consider maintaining an error pattern database that the decomposer can reference for known failure modes.
- Phase 2 requires a minimum number of resolved predictions before error patterns become meaningful.
