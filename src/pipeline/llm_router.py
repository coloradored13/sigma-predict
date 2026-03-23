"""Unified LLM routing layer.

Dispatches calls to Anthropic (local client), OpenAI, and Gemini
(via sigma-verify). Provides cross-model verification and challenge
capabilities through external providers only.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.config import Config

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Local Anthropic client matching sigma-verify's call() pattern.

    Does NOT have verify()/challenge() — Anthropic is the host model,
    not a verification target.
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
    ) -> tuple[str, int, int]:
        """General-purpose LLM call. Returns (text, tokens_in, tokens_out)."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self.model,
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

    Creates AnthropicClient locally and imports OpenAIClient/GeminiClient
    from sigma-verify. Routes call(), verify(), challenge(), and
    cross_verify() to the appropriate client.
    """

    def __init__(self, config: Config):
        self.config = config

        # Local Anthropic client (only if key available)
        self._anthropic = None
        if config.anthropic_api_key:
            self._anthropic = AnthropicClient(
                model="claude-sonnet-4-6",
                api_key=config.anthropic_api_key,
            )
            logger.info("Anthropic client available")

        # External clients from sigma-verify
        self._openai = None
        self._gemini = None

        try:
            from sigma_verify import OpenAIClient
            client = OpenAIClient()
            if client.available:
                self._openai = client
                logger.info("OpenAI client available: model=%s", client.model)
            else:
                logger.info("OpenAI client not available (no API key)")
        except ImportError:
            logger.warning("sigma-verify not installed — OpenAI client unavailable")

        try:
            from sigma_verify import GeminiClient
            client = GeminiClient()
            if client.available:
                self._gemini = client
                logger.info("Gemini client available: model=%s", client.model)
            else:
                logger.info("Gemini client not available (no API key)")
        except ImportError:
            logger.warning("sigma-verify not installed — Gemini client unavailable")

        logger.info(
            "LLMRouter ready: primary=%s, providers=%s",
            config.primary_model.provider if config.primary_model else "none",
            self.available_providers(),
        )

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
        """
        if provider == "anthropic":
            if self._anthropic is None:
                raise ValueError("Anthropic client not available (no API key)")
            old_model = self._anthropic.model
            self._anthropic.model = model
            try:
                return self._anthropic.call(system, user, temperature, max_tokens)
            finally:
                self._anthropic.model = old_model

        elif provider == "openai":
            if self._openai is None:
                raise ValueError("OpenAI client not available")
            old_model = self._openai.model
            self._openai.model = model
            try:
                return self._openai.call(system, user, temperature, max_tokens)
            finally:
                self._openai.model = old_model

        elif provider == "google":
            if self._gemini is None:
                raise ValueError("Gemini client not available")
            old_model = self._gemini.model
            self._gemini.model = model
            try:
                return self._gemini.call(system, user, temperature, max_tokens)
            finally:
                self._gemini.model = old_model

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def verify(
        self,
        finding: str,
        context: str,
        provider: str | None = None,
    ) -> Any | None:
        """Route verification to an external client.

        Returns VerificationResult on success, None on failure.
        External clients only (OpenAI/Gemini — Anthropic is the host model).
        """
        client = self._get_external_client(provider)
        if client is None:
            logger.debug("No external client available for verification")
            return None

        try:
            return client.verify(finding, context)
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

        try:
            result = client.challenge(claim, evidence)
            result["status"] = "success"
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
            try:
                result = client.verify(finding, context)
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

    def available_providers(self) -> list[str]:
        """Return list of provider names with valid API keys."""
        providers = []
        if self._anthropic is not None and self._anthropic.available:
            providers.append("anthropic")
        if self._openai is not None:
            providers.append("openai")
        if self._gemini is not None:
            providers.append("google")
        return providers

    def verification_available(self) -> bool:
        """Check if any external provider is available for verification."""
        return self._openai is not None or self._gemini is not None

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
        if provider == "openai":
            return self._openai
        if provider == "google":
            return self._gemini
        if provider is None:
            # Return first available external client
            if self._openai is not None:
                return self._openai
            if self._gemini is not None:
                return self._gemini
        return None

    def _external_clients(self) -> list:
        """Return list of clients available for verification.

        Excludes the primary provider — you shouldn't verify against the
        same model that generated the forecast.
        """
        primary = self.config.primary_model.provider if self.config.primary_model else None
        clients = []
        if self._openai is not None and primary != "openai":
            clients.append(self._openai)
        if self._gemini is not None and primary != "google":
            clients.append(self._gemini)
        # If the only available providers are the primary, include them anyway
        # (some verification is better than none)
        if not clients:
            if self._openai is not None:
                clients.append(self._openai)
            if self._gemini is not None:
                clients.append(self._gemini)
        return clients

    def _provider_name(self, client: Any) -> str:
        """Get the provider name for a client instance."""
        if client is self._openai:
            return "openai"
        if client is self._gemini:
            return "google"
        if client is self._anthropic:
            return "anthropic"
        return "unknown"
