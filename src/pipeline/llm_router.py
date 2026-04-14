"""Unified LLM routing layer.

Dispatches calls to Anthropic (local client), and all sigma-verify providers
(OpenAI, Gemini, and 10 Ollama-backed models). Provides cross-model
verification and challenge capabilities through external providers only.

Provider roles:
- Local AnthropicClient: primary forecasting calls only (host model).
  Not used for self-verification — Anthropic is the host model.
- sigma-verify PROVIDERS: external verification/challenge. When Anthropic
  is primary, all sigma-verify providers are used for verification.
  When another provider is primary, Anthropic (via sigma-verify) can join the
  verification pool if added to Config.verification_providers.
- 4B local models (nemotron-nano, qwen-local): verification only.
  Unreliable for structured JSON forecaster output.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.config import Config
from src.pipeline.llm_audit import LLMAuditLogger

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Local Anthropic client for primary forecasting calls.

    Does NOT have verify()/challenge() — Anthropic is the host model,
    not a verification target when used as primary.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or "claude-sonnet-4-6"
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def call(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> tuple[str, int, int]:
        """General-purpose LLM call. Returns (text, tokens_in, tokens_out)."""
        import anthropic

        resolved_model = model if model is not None else self.model
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )


class LLMRouter:
    """Unified dispatcher for all LLM calls.

    Creates AnthropicClient locally for primary forecasting and loads all
    sigma-verify PROVIDERS for external verification. Routes call(),
    verify(), challenge(), and cross_verify() to the appropriate client.

    Client cache: clients are keyed by (provider, model_id). No model
    mutation occurs — each (provider, model) pair gets its own instance.
    Add a threading.Lock around _client_cache access if concurrent calls
    are introduced.
    """

    def __init__(self, config: Config):
        self.config = config

        # Audit logger — None when config.audit_path is not set (zero overhead)
        self._audit: LLMAuditLogger | None = (
            LLMAuditLogger(config.audit_path) if config.audit_path else None
        )

        # Local Anthropic client for primary forecasting (host model)
        self._anthropic = None
        if config.anthropic_api_key:
            self._anthropic = AnthropicClient(
                model="claude-sonnet-4-6",
                api_key=config.anthropic_api_key,
            )
            logger.info("Anthropic client available (primary/host)")

        # Cache: (provider_name, model_id) -> client instance
        # Eliminates model-mutation tech debt — each call uses its own instance
        self._client_cache: dict[tuple[str, str], Any] = {}

        # External providers from sigma-verify PROVIDERS registry
        # verification_providers config controls which are active
        self._external_providers: dict[str, Any] = {}
        self._init_external_providers(config)

        logger.info(
            "LLMRouter ready: primary=%s, external_providers=%s",
            config.primary_model.provider if config.primary_model else "none",
            list(self._external_providers.keys()),
        )

    def _init_external_providers(self, config: Config) -> None:
        """Load external providers from sigma-verify PROVIDERS registry."""
        try:
            from sigma_verify import PROVIDERS
        except ImportError:
            logger.warning("sigma-verify not installed — no external providers available")
            return

        allowed = set(config.verification_providers)

        for name, cls in PROVIDERS.items():
            if name == "anthropic":
                # sigma-verify's AnthropicClient: only add when Anthropic is NOT primary
                # (avoids self-verification when Anthropic is the host model)
                if config.primary_model and config.primary_model.provider != "anthropic":
                    try:
                        client = cls()
                        if client.available and name in allowed:
                            self._external_providers[name] = client
                            logger.info("External provider available: %s model=%s", name, client.model)
                    except Exception as e:
                        logger.debug("Provider %s init failed: %s", name, e)
                continue

            if name not in allowed:
                continue

            try:
                client = cls()
                if client.available:
                    self._external_providers[name] = client
                    logger.info("External provider available: %s model=%s", name, client.model)
                else:
                    logger.debug("Provider %s not available (no API key/env)", name)
            except Exception as e:
                logger.debug("Provider %s init failed: %s", name, e)

        # Legacy: keep _openai and _gemini attributes for backward compatibility
        self._openai = self._external_providers.get("openai")
        self._gemini = self._external_providers.get("google")

    def _get_or_create_client(self, provider: str, model: str) -> Any:
        """Get a cached client for (provider, model), creating if needed.

        No model mutation: each (provider, model) pair has its own instance.
        """
        key = (provider, model)
        if key in self._client_cache:
            return self._client_cache[key]

        if provider == "anthropic":
            client = AnthropicClient(
                model=model,
                api_key=self.config.anthropic_api_key,
            )
        else:
            try:
                from sigma_verify import PROVIDERS
            except ImportError:
                raise ValueError("sigma-verify not installed")

            cls = PROVIDERS.get(provider)
            if cls is None:
                raise ValueError(f"Unknown provider: {provider}")

            client = cls(model=model)

        self._client_cache[key] = client
        return client

    def call(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """Dispatch an LLM call to the right client.

        Returns (response_text, tokens_in, tokens_out).
        No model mutation: uses per-(provider, model) cached client.
        """
        t0 = time.monotonic()
        if provider == "anthropic":
            if self._anthropic is None:
                raise ValueError("Anthropic client not available (no API key)")
            text, tokens_in, tokens_out = self._anthropic.call(system, user, temperature, max_tokens, model=model)
        else:
            # For non-anthropic providers, use cache to avoid model mutation
            client = self._get_or_create_client(provider, model)
            text, tokens_in, tokens_out = client.call(system, user, temperature, max_tokens)

        if self._audit is not None:
            duration_ms = (time.monotonic() - t0) * 1000
            cost = self.estimate_cost(provider, model, tokens_in, tokens_out)
            self._audit.log_call(
                provider=provider,
                model=model,
                method="call",
                params={"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens},
                result_text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=duration_ms,
                cost_usd=cost,
            )

        return text, tokens_in, tokens_out

    def verify(
        self,
        finding: str,
        context: str,
        provider: str | None = None,
    ) -> Any | None:
        """Route verification to an external client.

        Returns VerificationResult on success, None on failure.
        External clients only (Anthropic is the host model).
        """
        client = self._get_external_client(provider)
        if client is None:
            logger.debug("No external client available for verification")
            return None

        t0 = time.monotonic()
        try:
            result = client.verify(finding, context)
            if self._audit is not None:
                duration_ms = (time.monotonic() - t0) * 1000
                provider_name = self._provider_name(client)
                result_text = str(result) if result is not None else ""
                self._audit.log_call(
                    provider=provider_name,
                    model=client.model,
                    method="verify",
                    params={"finding": finding, "context": context},
                    result_text=result_text,
                    tokens_in=0,
                    tokens_out=0,
                    duration_ms=duration_ms,
                )
            return result
        except Exception as e:
            logger.warning("Verification failed (%s): %s", type(e).__name__, e)
            try:
                from sigma_verify import classify_error, VerificationError
                return VerificationError(
                    model=client.model,
                    provider=self._provider_name(client),
                    error_class=classify_error(e),
                    error_detail=str(e),
                    tool_attempted="verify_finding",
                    finding_brief=finding[:100],
                    provenance_tag=f"|source:external-{self._provider_name(client)}-{client.model}|",
                )
            except ImportError:
                return None

    def challenge(
        self,
        claim: str,
        evidence: str,
        provider: str | None = None,
    ) -> dict | None:
        """Route challenge to an external client.

        Returns challenge dict on success, None on failure.
        External clients only.
        """
        client = self._get_external_client(provider)
        if client is None:
            logger.debug("No external client available for challenge")
            return None

        t0 = time.monotonic()
        try:
            result = client.challenge(claim, evidence)
            result["status"] = "success"
            if self._audit is not None:
                duration_ms = (time.monotonic() - t0) * 1000
                provider_name = self._provider_name(client)
                self._audit.log_call(
                    provider=provider_name,
                    model=client.model,
                    method="challenge",
                    params={"claim": claim, "evidence": evidence},
                    result_text=str(result),
                    tokens_in=0,
                    tokens_out=0,
                    duration_ms=duration_ms,
                )
            return result
        except Exception as e:
            logger.warning("Challenge failed (%s): %s", type(e).__name__, e)
            return {
                "provider": self._provider_name(client),
                "model": client.model,
                "status": "failed",
                "error": str(e),
            }

    def cross_verify(
        self,
        finding: str,
        context: str,
    ) -> list:
        """Call all available external clients for verification.

        Returns list of VerificationResult/VerificationError objects.
        """
        results = []
        for client in self._external_clients():
            t0 = time.monotonic()
            try:
                result = client.verify(finding, context)
                if self._audit is not None:
                    duration_ms = (time.monotonic() - t0) * 1000
                    provider_name = self._provider_name(client)
                    result_text = str(result) if result is not None else ""
                    self._audit.log_call(
                        provider=provider_name,
                        model=client.model,
                        method="cross_verify",
                        params={"finding": finding, "context": context},
                        result_text=result_text,
                        tokens_in=0,
                        tokens_out=0,
                        duration_ms=duration_ms,
                    )
                results.append(result)
            except Exception as e:
                logger.warning(
                    "Cross-verify failed for %s: %s",
                    self._provider_name(client), e,
                )
                try:
                    from sigma_verify import classify_error, VerificationError
                    results.append(VerificationError(
                        model=client.model,
                        provider=self._provider_name(client),
                        error_class=classify_error(e),
                        error_detail=str(e),
                        tool_attempted="cross_verify",
                        finding_brief=finding[:100],
                        provenance_tag=f"|source:external-{self._provider_name(client)}-{client.model}|",
                    ))
                except ImportError:
                    pass
        return results

    def flush_audit(self) -> None:
        """Flush buffered audit log entries to disk."""
        if self._audit is not None:
            self._audit.flush()

    def available_providers(self) -> list[str]:
        """Return list of provider names with valid API keys."""
        providers = []
        if self._anthropic is not None and self._anthropic.available:
            providers.append("anthropic")
        providers.extend(self._external_providers.keys())
        return providers

    def verification_available(self) -> bool:
        """Check if any external provider is available for verification."""
        return bool(self._external_providers)

    def estimate_cost(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> float:
        """Estimate USD cost for an API call."""
        # Pricing per million tokens (approximate, March 2026)
        pricing = {
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-opus-4-6": (5.0, 25.0),
            "claude-haiku-4-5": (1.0, 5.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-5.1": (3.0, 12.0),
            "o4-mini": (1.10, 4.40),
            "gemini-3.1": (1.25, 5.0),
            "gemini-2.5-pro": (1.25, 10.0),
            "gemini-2.5-flash": (0.15, 0.60),
        }
        rates = pricing.get(model, (3.0, 15.0))  # default to Sonnet pricing
        cost = (tokens_in / 1_000_000 * rates[0]) + (tokens_out / 1_000_000 * rates[1])
        return round(cost, 6)

    def _get_external_client(self, provider: str | None = None) -> Any | None:
        """Get an external client by provider name, or the first available."""
        if provider is not None:
            return self._external_providers.get(provider)
        # Return first available external client
        if self._external_providers:
            return next(iter(self._external_providers.values()))
        return None

    def _external_clients(self) -> list:
        """Return list of clients available for verification.

        Excludes the primary provider — you shouldn't verify against the
        same model that generated the forecast.
        """
        primary = self.config.primary_model.provider if self.config.primary_model else None
        clients = [
            client for name, client in self._external_providers.items()
            if name != primary
        ]
        # If the only available providers are the primary, include them anyway
        if not clients:
            clients = list(self._external_providers.values())
        return clients

    def _provider_name(self, client: Any) -> str:
        """Get the provider name for a client instance."""
        for name, c in self._external_providers.items():
            if c is client:
                return name
        if client is self._anthropic:
            return "anthropic"
        return "unknown"

