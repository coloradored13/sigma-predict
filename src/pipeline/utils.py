"""Shared pipeline utilities — extracted from decomposer, base_rate, forecaster."""

from __future__ import annotations

import os

from src.config import Config


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


def _load_prompt(config: Config, filename: str) -> str:
    path = os.path.join(config.prompts_dir, filename)
    with open(path) as f:
        return f.read()
