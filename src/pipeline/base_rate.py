"""Isolated base rate estimation.

CRITICAL DESIGN: This runs as a SEPARATE LLM call BEFORE any inside-view
search. The base rate is computed in isolation to prevent anchoring on
compelling current evidence. The output is locked and fed as a fixed
anchor to all estimation runs.
"""

from __future__ import annotations

import json
import logging

from src.config import Config
from src.pipeline.utils import _extract_json, _load_prompt

logger = logging.getLogger(__name__)


def estimate_base_rate(
    question_text: str,
    sub_questions: list[str],
    reference_class: str,
    config: Config,
    router=None,
    warnings_out: list[str] | None = None,
) -> tuple[float, str, str]:
    """Estimate the base rate for a question in isolation.

    Returns:
        tuple of (base_rate, reference_class, source)
    """
    if router is None:
        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter(config)

    system_prompt = _load_prompt(config, "base_rate.md")

    user_content = f"""## Question
{question_text}

## Sub-Questions (from decomposition)
{chr(10).join(f'- {sq}' for sq in sub_questions)}

## Initial Reference Class (from decomposition)
{reference_class}

Produce your base rate estimate now. Remember: use ONLY historical/statistical reasoning.
Do NOT consider current events, recent news, or inside-view evidence.
"""

    raw, _, _ = router.call(
        config.primary_model.provider,
        config.primary_model.model_id,
        system_prompt,
        user_content,
        temperature=0.2,
        max_tokens=1024,
    )
    json_str = _extract_json(raw)

    try:
        data = json.loads(json_str)
        base_rate = float(data.get("base_rate", 0.5))
        ref_class = data.get("reference_class", reference_class)
        source = data.get("base_rate_source", "LLM estimate")

        # Clamp to valid range
        base_rate = max(0.01, min(0.99, base_rate))

        logger.info(
            "Base rate locked: %.2f (reference class: %s)",
            base_rate, ref_class,
        )
        return base_rate, ref_class, source

    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse base rate response, defaulting to 0.5")
        logger.debug("Raw response: %s", raw)
        if warnings_out is not None:
            warnings_out.append("base_rate_parse_failure")
        return 0.5, reference_class, "parse_failure_default"


def search_augmented_base_rate(
    base_rate: float,
    reference_class: str,
    search_module,
    router,
    config: Config,
) -> tuple[float, str]:
    """Augment the base rate estimate with statistical-only web search.

    Contamination controls:
    1. Restricted query: "{reference_class} historical frequency statistics" ONLY
    2. Temporal firewall: exclude results <6 months old (client-side date check)
    3. Separate model call: only numerical rate + citation fed in (no inside-view context)
    4. Output labeled: STATISTICAL REFERENCE DATA

    Called AFTER the base LLM estimate. The result is a statistical anchor that
    can narrow the reference class estimate — it does NOT replace the LLM estimate
    but provides an empirical check.

    Args:
        base_rate: LLM base rate estimate (from estimate_base_rate())
        reference_class: Reference class string (e.g., "coup attempts in MENA")
        search_module: SearchModule instance
        router: LLMRouter instance
        config: Pipeline config

    Returns:
        tuple of (augmented_rate, citation_note)
        - augmented_rate: float in [0.01, 0.99]
        - citation_note: string describing the statistical source (or empty string on failure)
    """
    from datetime import date, timedelta

    # Restricted query — only historical statistics, never current events
    query = f"{reference_class} historical frequency statistics"

    try:
        results = search_module.search(query)
    except Exception as e:
        logger.warning("search_augmented_base_rate: search failed: %s", e)
        return base_rate, ""

    if not results:
        logger.info("search_augmented_base_rate: no search results, returning base rate unchanged")
        return base_rate, ""

    # Temporal firewall: exclude results published within the last 6 months
    cutoff = date.today() - timedelta(days=180)
    filtered_results = []
    for r in results:
        pub_date_str = getattr(r, "published_date", None) or getattr(r, "date", None)
        if pub_date_str:
            try:
                # Parse common ISO-8601 variants
                pub_str = str(pub_date_str)[:10]  # take YYYY-MM-DD prefix
                pub_date = date.fromisoformat(pub_str)
                if pub_date >= cutoff:
                    logger.debug(
                        "search_augmented_base_rate: excluding recent result (date=%s): %s",
                        pub_date_str, getattr(r, "title", ""),
                    )
                    continue
            except (ValueError, TypeError):
                # If date unparseable, include with warning flag
                logger.debug(
                    "search_augmented_base_rate: unparseable date '%s', including with flag",
                    pub_date_str,
                )
        filtered_results.append(r)

    if not filtered_results:
        logger.info(
            "search_augmented_base_rate: all results excluded by temporal firewall, "
            "returning base rate unchanged"
        )
        return base_rate, ""

    # Build a minimal, contamination-safe context: only numerical rates and citations
    # Do NOT include content that could anchor inside-view reasoning
    stat_snippets = []
    for r in filtered_results[:3]:  # limit to 3 most relevant
        title = getattr(r, "title", "")
        url = getattr(r, "url", "")
        content = getattr(r, "content", "")[:300]
        stat_snippets.append(f"Source: {title} ({url})\nExcerpt: {content}")

    stat_context = "\n\n".join(stat_snippets)

    # Separate model call: feed only statistical context
    # No original probability, no reasoning, no current events
    system_prompt = (
        "You are a statistical reference extractor. Given search results about historical "
        "base rates for a reference class, extract the most relevant numerical rate. "
        "Respond with JSON only: "
        '{"statistical_rate": <float 0.01-0.99>, "citation": "<source name>", '
        '"confidence": "high" | "medium" | "low"}. '
        "If no clear numerical rate is present, respond: "
        '{"statistical_rate": null, "citation": "", "confidence": "low"}.'
    )

    user_content = (
        f"STATISTICAL REFERENCE DATA\n\n"
        f"Reference class: {reference_class}\n\n"
        f"Search results (historical only — temporal firewall applied):\n\n"
        f"{stat_context}\n\n"
        f"Extract the numerical historical frequency for this reference class."
    )

    try:
        raw, _, _ = router.call(
            config.primary_model.provider,
            config.primary_model.model_id,
            system_prompt,
            user_content,
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as e:
        logger.warning("search_augmented_base_rate: LLM call failed: %s", e)
        return base_rate, ""

    json_str = _extract_json(raw)
    try:
        stat_data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("search_augmented_base_rate: failed to parse LLM response")
        return base_rate, ""

    stat_rate = stat_data.get("statistical_rate")
    citation = stat_data.get("citation", "")
    confidence = stat_data.get("confidence", "low")

    if stat_rate is None:
        logger.info(
            "search_augmented_base_rate: no numerical rate extracted, "
            "returning base rate unchanged"
        )
        return base_rate, ""

    stat_rate = float(stat_rate)
    stat_rate = max(0.01, min(0.99, stat_rate))

    if confidence == "low":
        # Low-confidence stat: blend conservatively (25% weight on stat)
        augmented = 0.75 * base_rate + 0.25 * stat_rate
    elif confidence == "medium":
        # Medium confidence: 40% weight on stat
        augmented = 0.60 * base_rate + 0.40 * stat_rate
    else:
        # High confidence: 50% weight (equal blend)
        augmented = 0.50 * base_rate + 0.50 * stat_rate

    augmented = max(0.01, min(0.99, augmented))

    citation_note = (
        f"STATISTICAL REFERENCE DATA: {stat_rate:.3f} from '{citation}' "
        f"(confidence={confidence}, blended with LLM estimate {base_rate:.3f})"
    )

    logger.info(
        "search_augmented_base_rate: base=%.3f, stat=%.3f (confidence=%s), "
        "augmented=%.3f — %s",
        base_rate, stat_rate, confidence, augmented, citation,
    )

    return augmented, citation_note
