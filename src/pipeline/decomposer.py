"""Question decomposition and actor motivation analysis."""

from __future__ import annotations

import json
import logging

from src.config import Config
from src.models import (
    Actor,
    ActorAnalysis,
    ActorLevel,
    ACTOR_LEVEL_LABELS,
    Decomposition,
    PeripheralActor,
    PeripheralCategory,
    PlatformQuestion,
)
from src.pipeline.utils import _extract_json, _load_prompt

logger = logging.getLogger(__name__)


def decompose_question(
    question: PlatformQuestion,
    config: Config,
    router=None,
    warnings_out: list[str] | None = None,
) -> Decomposition:
    """Decompose a question into sub-questions and analyze actor motivations.

    This is the first pipeline stage. It produces:
    - Fermi sub-questions for structured analysis
    - Actor classification using the 3-level motivation hierarchy
    - Probability space constraints
    """
    if router is None:
        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter(config)

    system_prompt = _load_prompt(config, "decomposer.md")

    user_content = f"""## Question to Decompose

**Title:** {question.title}

**Description:** {question.description}

**Resolution Criteria:** {question.resolution_criteria}

**Fine Print:** {question.fine_print}

**Question Type:** {question.question_type.value}

**Close Date:** {question.close_date}
**Resolve Date:** {question.resolve_date}

**Tags:** {', '.join(question.tags)}
"""

    raw, _, _ = router.call(
        config.primary_model.provider,
        config.primary_model.model_id,
        system_prompt,
        user_content,
        temperature=0.3,
        max_tokens=config.primary_model.max_tokens,
    )

    # Extract JSON from response (may be wrapped in markdown code block)
    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse decomposition JSON, using defaults")
        logger.debug("Raw response: %s", raw)
        if warnings_out is not None:
            warnings_out.append("decomposition_parse_failure")
        return Decomposition()

    # Build Decomposition from parsed data
    actors = []
    actor_data = data.get("actor_analysis", {})
    for a in actor_data.get("actors", []):
        level = a.get("level", 1)
        actors.append(Actor(
            name=a.get("name", ""),
            level=ActorLevel(level),
            level_label=ACTOR_LEVEL_LABELS.get(ActorLevel(level), "convergent_rational"),
            motivation=a.get("motivation", ""),
            influence_estimate=a.get("influence_estimate", "low"),
            observable_indicators=a.get("observable_indicators", []),
            implication_for_estimate=a.get("implication_for_estimate", ""),
        ))

    has_level_23 = any(a.level in (ActorLevel.IDENTITY_CONSTRAINED, ActorLevel.DIVERGENT_PREFERENCE) for a in actors)

    # Parse peripheral actors
    peripheral_actors = []
    category_map = {
        "beneficiary": PeripheralCategory.BENEFICIARY,
        "collateral": PeripheralCategory.COLLATERAL_ENTRANT,
        "collateral_entrant": PeripheralCategory.COLLATERAL_ENTRANT,
        "leverage_holder": PeripheralCategory.LEVERAGE_HOLDER,
        "opportunistic": PeripheralCategory.OPPORTUNISTIC,
    }
    for pa in actor_data.get("peripheral_actors", []):
        cats = []
        for c in pa.get("categories", []):
            mapped = category_map.get(c)
            if mapped:
                cats.append(mapped)
        peripheral_actors.append(PeripheralActor(
            name=pa.get("name", ""),
            categories=cats,
            motivation=pa.get("motivation", ""),
            leverage=pa.get("leverage", ""),
            feedback_mechanism=pa.get("feedback_mechanism", ""),
            threshold_for_action=pa.get("threshold_for_action", ""),
            influence_estimate=pa.get("influence_estimate", "low"),
        ))

    decomposition = Decomposition(
        sub_questions=data.get("sub_questions", []),
        reference_class=data.get("reference_class", ""),
        base_rate=data.get("base_rate", 0.5),
        base_rate_source=data.get("base_rate_source", ""),
        actor_analysis=ActorAnalysis(
            actors_involved=actor_data.get("actors_involved", len(actors) > 0),
            actors=actors,
            level_2_3_present=has_level_23,
            peripheral_actors=peripheral_actors,
            probability_space_constraints=actor_data.get(
                "probability_space_constraints", []
            ),
        ),
    )

    logger.info(
        "Decomposition complete: %d sub-questions, %d actors identified (%s Level 2/3)",
        len(decomposition.sub_questions),
        len(actors),
        "has" if has_level_23 else "no",
    )

    return decomposition
