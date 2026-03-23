"""Single-run forecaster.

Each run is an independent estimation: it receives the question, decomposition,
locked base rate, and its own search results. It produces a probability
estimate with full reasoning chain.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

from src.config import Config
from src.models import (
    CostTracking,
    Decomposition,
    PlatformQuestion,
    RunResult,
    SearchPersona,
)

logger = logging.getLogger(__name__)


def _load_prompt(config: Config, filename: str) -> str:
    path = os.path.join(config.prompts_dir, filename)
    with open(path) as f:
        return f.read()


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown code blocks."""
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    text = text.strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def _format_search_results(search_results: list) -> str:
    """Format search results for inclusion in prompt."""
    if not search_results:
        return "No search results available."

    parts = []
    for i, r in enumerate(search_results, 1):
        parts.append(f"### Source {i}: {r.title}")
        parts.append(f"**URL:** {r.url}")
        parts.append(f"**Relevance Score:** {r.score:.2f}")
        parts.append(f"**Content:** {r.content}")
        parts.append("")
    return "\n".join(parts)


def _format_actor_analysis(decomposition: Decomposition) -> str:
    """Format actor analysis for inclusion in prompt."""
    aa = decomposition.actor_analysis
    if not aa.actors_involved:
        return "No actors directly involved in this outcome."

    parts = ["## Actor Analysis (from decomposition)\n"]
    for actor in aa.actors:
        parts.append(f"### {actor.name}")
        parts.append(f"- **Level:** {actor.level} ({actor.level_label})")
        parts.append(f"- **Motivation:** {actor.motivation}")
        parts.append(f"- **Influence:** {actor.influence_estimate}")
        if actor.observable_indicators:
            parts.append(f"- **Observable indicators:** {', '.join(actor.observable_indicators)}")
        if actor.implication_for_estimate:
            parts.append(f"- **Implication:** {actor.implication_for_estimate}")
        parts.append("")

    if aa.peripheral_actors:
        parts.append("\n### Peripheral Actors (feedback loops)")
        category_labels = {
            "beneficiary": "Beneficiary of Continuation",
            "collateral": "Collateral Entrant",
            "leverage_holder": "Leverage Holder",
            "opportunistic": "Opportunistic Repositioner",
        }
        for pa in aa.peripheral_actors:
            cat_str = ", ".join(category_labels.get(c.value, c.value) for c in pa.categories)
            parts.append(f"\n**{pa.name}** [{cat_str}]")
            if pa.motivation:
                parts.append(f"- Motivation: {pa.motivation}")
            if pa.leverage:
                parts.append(f"- Leverage: {pa.leverage}")
            if pa.feedback_mechanism:
                parts.append(f"- Feedback mechanism: {pa.feedback_mechanism}")
            if pa.threshold_for_action:
                parts.append(f"- Threshold for action: {pa.threshold_for_action}")

    if aa.probability_space_constraints:
        parts.append("\n### Probability Space Constraints")
        for c in aa.probability_space_constraints:
            parts.append(f"- {c}")

    return "\n".join(parts)


def run_forecast(
    question: PlatformQuestion,
    decomposition: Decomposition,
    base_rate: float,
    base_rate_source: str,
    search_results: list,
    persona: SearchPersona,
    config: Config,
    model_id: str | None = None,
    provider: str | None = None,
    router=None,
) -> tuple[RunResult, CostTracking]:
    """Execute a single forecasting run.

    This is the core estimation unit. Each call is independent — different
    search results, potentially different models.

    Returns:
        tuple of (RunResult, CostTracking)
    """
    start_time = time.time()
    model = model_id or config.primary_model.model_id
    prov = provider or config.primary_model.provider

    if router is None:
        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter(config)

    # Load prompts
    forecaster_prompt = _load_prompt(config, "forecaster_base.md")
    pre_mortem_prompt = _load_prompt(config, "pre_mortem.md")

    # Load persona prompt
    persona_path = f"personas/{persona.value}.md"
    try:
        persona_prompt = _load_prompt(config, persona_path)
    except FileNotFoundError:
        persona_prompt = ""

    # Build the full system prompt
    system = f"{forecaster_prompt}\n\n## Search Persona\n{persona_prompt}"

    # Build user content
    user_content = f"""## Question
**Title:** {question.title}
**Description:** {question.description}
**Resolution Criteria:** {question.resolution_criteria}
**Type:** {question.question_type.value}
**Close Date:** {question.close_date}
**Resolve Date:** {question.resolve_date}

## Decomposition
**Sub-questions:**
{chr(10).join(f'- {sq}' for sq in decomposition.sub_questions)}

## Base Rate (LOCKED — computed in isolation before search)
**Base rate:** {base_rate:.3f}
**Reference class:** {decomposition.reference_class}
**Source:** {base_rate_source}

{_format_actor_analysis(decomposition)}

## Search Results
{_format_search_results(search_results)}

Produce your forecast now. Start from the base rate of {base_rate:.3f} and adjust based on the evidence above.
"""

    total_tokens_in = 0
    total_tokens_out = 0

    # Main forecast call
    result, t_in, t_out = router.call(
        prov, model, system, user_content,
        temperature=config.primary_model.temperature,
        max_tokens=config.primary_model.max_tokens,
    )

    total_tokens_in += t_in
    total_tokens_out += t_out

    # Parse forecast result
    json_str = _extract_json(result)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse forecast JSON from %s, using base rate", model)
        logger.debug("Raw response: %s", result)
        data = {}

    probability = float(data.get("probability", base_rate))
    probability = max(0.01, min(0.99, probability))
    inside_view_adj = probability - base_rate

    # Pre-mortem step
    pre_mortem_user = f"""## Original Forecast
**Question:** {question.title}
**Your probability estimate:** {probability:.3f}
**Your reasoning:** {data.get('reasoning_chain', 'Not provided')}

{_format_actor_analysis(decomposition)}

Now run your pre-mortem analysis. Imagine your prediction was WRONG.
"""

    pm_result, pm_in, pm_out = router.call(
        prov, model, pre_mortem_prompt, pre_mortem_user,
        temperature=config.primary_model.temperature,
        max_tokens=config.primary_model.max_tokens,
    )

    total_tokens_in += pm_in
    total_tokens_out += pm_out

    pm_json = _extract_json(pm_result)
    try:
        pm_data = json.loads(pm_json)
    except json.JSONDecodeError:
        pm_data = {}

    # Apply pre-mortem adjustment
    pm_changed = pm_data.get("pre_mortem_changed_estimate", False)
    pm_delta = float(pm_data.get("pre_mortem_delta", 0.0))
    if pm_changed:
        probability = max(0.01, min(0.99, probability + pm_delta))

    elapsed = time.time() - start_time

    run_result = RunResult(
        run_id=str(uuid.uuid4()),
        model=model,
        search_persona=persona.value,
        search_queries=data.get("search_queries_used", []),
        sources_cited=data.get("sources_cited", []),
        search_quality_score=int(data.get("search_quality_score", 5)),
        base_rate_used=base_rate,
        inside_view_adjustment=inside_view_adj,
        adjustment_reasoning=data.get("adjustment_reasoning", ""),
        pre_mortem_scenario=pm_data.get("pre_mortem_scenario", ""),
        pre_mortem_changed_estimate=pm_changed,
        pre_mortem_delta=pm_delta,
        actor_motivation_flags=pm_data.get("actor_motivation_flags", []),
        probability=probability,
        confidence_self_score=int(data.get("confidence_self_score", 5)),
        reasoning_chain=data.get("reasoning_chain", ""),
    )

    cost = CostTracking(
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_cost_usd=router.estimate_cost(prov, model, total_tokens_in, total_tokens_out),
        latency_seconds=elapsed,
    )

    logger.info(
        "Forecast run complete: model=%s persona=%s probability=%.3f (base=%.3f, adj=%+.3f, pm=%+.3f)",
        model, persona.value, probability, base_rate, inside_view_adj, pm_delta,
    )

    return run_result, cost


