"""Lightweight audit logger for LLM calls.

Append-only JSON-L format. Each LLM call is logged with:
timestamp, provider, model, method, params_hash, result_hash,
duration_ms, tokens_in, tokens_out, cost_usd.

Security: raw prompt/system text is NEVER written to the log.
Parameters are represented as a structural summary (types and lengths).
Set Config.audit_path to enable; None = disabled (zero overhead).

Ported from ollama-mcp-bridge audit pattern — simplified for sigma-predict's
single-threaded, personal-tool context.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Keys whose values are never included in structural summaries
_SECRET_KEY_PATTERN = {"api_key", "apikey", "token", "secret", "password", "auth"}


def _structural_summary(params: dict[str, Any]) -> str:
    """Build a structural summary of params — types/lengths, no raw values."""
    parts: list[str] = []
    for key, value in params.items():
        if any(s in key.lower() for s in _SECRET_KEY_PATTERN):
            parts.append(f"{key}:[REDACTED]")
        elif isinstance(value, str):
            parts.append(f"{key}:str({len(value)})")
        elif isinstance(value, (int, float, bool)):
            parts.append(f"{key}:{type(value).__name__}")
        elif isinstance(value, list):
            parts.append(f"{key}:list({len(value)})")
        elif isinstance(value, dict):
            parts.append(f"{key}:dict({len(value)})")
        elif value is None:
            parts.append(f"{key}:null")
        else:
            parts.append(f"{key}:{type(value).__name__}")
    return "{" + ", ".join(parts) + "}"


def _hash(data: str) -> str:
    """SHA-256 of data, truncated to 16 hex chars for readability."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class LLMAuditLogger:
    """Per-call audit logger for LLMRouter.

    Buffers entries and flushes periodically. Flush is also called
    on explicit flush() or when buffer reaches limit.

    Usage:
        audit = LLMAuditLogger(config.audit_path)
        audit.log_call(provider, model, method, params, result_text,
                       tokens_in, tokens_out, duration_ms, cost_usd)
        audit.flush()  # at pipeline end
    """

    BUFFER_LIMIT = 10

    def __init__(self, audit_path: str):
        self._path = Path(audit_path).expanduser()
        self._buffer: list[dict] = []
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_call(
        self,
        provider: str,
        model: str,
        method: str,
        params: dict[str, Any],
        result_text: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        cost_usd: float = 0.0,
    ) -> None:
        """Log a single LLM call."""
        params_summary = _structural_summary(params)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "method": method,
            "params_hash": _hash(params_summary),
            "params_summary": params_summary,
            "result_hash": _hash(result_text) if result_text else "",
            "result_len": len(result_text),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration_ms": round(duration_ms, 1),
            "cost_usd": round(cost_usd, 6),
        }
        self._buffer.append(entry)
        if len(self._buffer) >= self.BUFFER_LIMIT:
            self.flush()

    def flush(self) -> None:
        """Write buffered entries to disk."""
        if not self._buffer:
            return
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("LLMAuditLogger flush failed: %s", e)
        finally:
            self._buffer.clear()
