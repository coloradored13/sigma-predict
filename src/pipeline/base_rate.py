"""Isolated base rate estimation.

CRITICAL DESIGN: This runs as a SEPARATE LLM call BEFORE any inside-view
search. The base rate is computed in isolation to prevent anchoring on
compelling current evidence. The output is locked and fed as a fixed
anchor to all estimation runs.
"""

from __future__ import annotations

import json
import logging
import os

from src.config import Config
from src.models import Decomposition

logger = logging.getLogger(__name__)


def _load_prompt(config: Config) -> str:
    path = os.path.join(config.prompts_dir, "base_rate.md")
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


def estimate_base_rate(
    question_text: str,
    sub_questions: list[str],
    reference_class: str,
    config: Config,
    router=None,
) -> tuple[float, str, str]:
    """Estimate the base rate for a question in isolation.

    Returns:
        tuple of (base_rate, reference_class, source)
    """
    if router is None:
        from src.pipeline.llm_router import LLMRouter
        router = LLMRouter(config)

    system_prompt = _load_prompt(config)

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
        "anthropic",
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
        return 0.5, reference_class, "parse_failure_default"
