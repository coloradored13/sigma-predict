"""Configuration for sigma-predict pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (sigma-predict/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class ModelConfig:
    provider: str  # "anthropic" | "openai" | "google"
    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class SearchConfig:
    backend: str = "tavily"  # "tavily" | "serper" | "exa"
    max_results: int = 5
    search_depth: str = "advanced"


@dataclass
class AggregationConfig:
    method: str = "extremized_trimmed_mean"
    trim_fraction: float = 0.1
    extremization_factor: float = 1.5  # push away from 0.5
    deliberation_stdev_threshold: float = 0.15


@dataclass
class CalibrationConfig:
    prior_hedging_shift: float = 0.04  # shift away from 50% before Platt data exists
    platt_refit_interval: int = 50  # refit every N resolved predictions
    min_resolved_for_platt: int = 50


@dataclass
class VerificationConfig:
    enabled: bool = True
    verify_base_rate: bool = True
    verify_actors: bool = True
    verify_forecasts: str = "outliers"  # "all" | "outliers" | "none"
    outlier_threshold: float = 0.15
    challenge_pre_mortem: bool = True


@dataclass
class Config:
    # API keys (from environment)
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    google_ai_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_AI_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))
    metaculus_api_key: str = field(default_factory=lambda: os.environ.get("METACULUS_API_KEY", ""))
    polymarket_api_key: str = field(default_factory=lambda: os.environ.get("POLYMARKET_API_KEY", ""))
    kalshi_api_key: str = field(default_factory=lambda: os.environ.get("KALSHI_API_KEY", ""))

    # Model configs — auto-detect primary based on available keys
    primary_model: ModelConfig = field(default=None)

    # Available models for multi-model runs (Phase 1)
    models: list[ModelConfig] = field(default_factory=list)

    def __post_init__(self):
        """Auto-detect primary model and available models from API keys."""
        if self.primary_model is None:
            if self.anthropic_api_key:
                self.primary_model = ModelConfig(
                    provider="anthropic", model_id="claude-sonnet-4-6",
                    max_tokens=4096, temperature=0.7,
                )
            elif self.openai_api_key:
                self.primary_model = ModelConfig(
                    provider="openai", model_id="gpt-5.1",
                    max_tokens=4096, temperature=0.7,
                )
            elif self.google_ai_api_key:
                self.primary_model = ModelConfig(
                    provider="google", model_id="gemini-3.1",
                    max_tokens=4096, temperature=0.7,
                )

        if not self.models:
            if self.anthropic_api_key:
                self.models.append(ModelConfig(provider="anthropic", model_id="claude-sonnet-4-6"))
            if self.openai_api_key:
                self.models.append(ModelConfig(provider="openai", model_id="gpt-5.1"))
            if self.google_ai_api_key:
                self.models.append(ModelConfig(provider="google", model_id="gemini-3.1"))

    # Search
    search: SearchConfig = field(default_factory=SearchConfig)

    # Aggregation
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    # Calibration
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Verification (cross-model)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    # Paths
    data_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    ))
    prompts_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
    ))

    @property
    def predictions_dir(self) -> str:
        return os.path.join(self.data_dir, "predictions")

    @property
    def calibration_dir(self) -> str:
        return os.path.join(self.data_dir, "calibration")

    @property
    def registry_path(self) -> str:
        return os.path.join(self.predictions_dir, "registry.jsonl")

    def validate(self) -> list[str]:
        """Return list of missing required config items."""
        missing = []
        # At least one LLM provider is required
        if not any([self.anthropic_api_key, self.openai_api_key, self.google_ai_api_key]):
            missing.append("ANTHROPIC_API_KEY or OPENAI_API_KEY or GOOGLE_AI_API_KEY (at least one)")
        if self.primary_model is None:
            missing.append("No LLM provider available (need at least one API key)")
        return missing
