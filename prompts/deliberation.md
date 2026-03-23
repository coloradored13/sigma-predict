# Supervisor Deliberation — Phase 3

> **Status**: Placeholder for Phase 3 implementation.

This prompt will drive the supervisor deliberation process, triggered when multiple forecasting runs produce estimates with high disagreement (standard deviation exceeding the configured threshold).

## Planned Functionality

The deliberation supervisor will receive:
- All individual run results (probabilities, reasoning chains, evidence cited)
- The aggregated estimate and disagreement metrics
- Actor analysis from the decomposition step

The supervisor will:
1. Identify the crux of disagreement — which evidence or reasoning step drives the divergence
2. Evaluate which run(s) have stronger evidentiary support
3. Determine if the disagreement reflects genuine uncertainty or an error in one run
4. Produce a deliberated estimate that resolves the disagreement with explicit justification
5. Flag whether the question should be escalated for human review

## Output Schema (Draft)

```json
{
  "deliberation_triggered_by": "stdev_threshold | cross_model_disagreement",
  "crux_of_disagreement": "What the runs disagree about",
  "resolution_reasoning": "How the supervisor resolved the disagreement",
  "deliberated_probability": 0.0,
  "confidence_in_deliberation": 0,
  "escalate_to_human": false,
  "escalation_reason": ""
}
```

## Implementation Notes

- This will require a more capable model (or the same model with a richer context window) to serve as supervisor.
- The deliberation prompt must have access to the full reasoning chains, not just final probabilities.
- Consider using structured debate format: each run "argues" for its estimate, supervisor adjudicates.
