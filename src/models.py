"""Core data models for the prediction registry."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Platform(str, Enum):
    METACULUS = "metaculus"
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class QuestionType(str, Enum):
    BINARY = "binary"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"


class ActorLevel(int, Enum):
    CONVERGENT_RATIONAL = 1
    IDENTITY_CONSTRAINED = 2
    DIVERGENT_PREFERENCE = 3


ACTOR_LEVEL_LABELS = {
    ActorLevel.CONVERGENT_RATIONAL: "convergent_rational",
    ActorLevel.IDENTITY_CONSTRAINED: "identity_constrained",
    ActorLevel.DIVERGENT_PREFERENCE: "divergent_preference",
}


class SearchPersona(str, Enum):
    ACADEMIC = "academic"
    NEWS = "news"
    CONTRARIAN = "contrarian"
    DOMAIN_EXPERT = "domain_expert"
    META_FORECASTER = "meta_forecaster"


class PeripheralCategory(str, Enum):
    """Categories for peripheral actors based on relationship to outcome."""
    BENEFICIARY = "beneficiary"           # Category A: profits from continuation
    COLLATERAL_ENTRANT = "collateral"     # Category B: dragged in involuntarily
    LEVERAGE_HOLDER = "leverage_holder"   # Category C: could change trajectory
    OPPORTUNISTIC = "opportunistic"       # Category D: using distraction for own agenda


class Actor(BaseModel):
    name: str
    level: ActorLevel
    level_label: str
    motivation: str
    influence_estimate: Literal["low", "medium", "high"] = "low"
    observable_indicators: list[str] = Field(default_factory=list)
    implication_for_estimate: str = ""


class PeripheralActor(BaseModel):
    """Actor not driving the outcome but creating feedback loops that affect it."""
    name: str
    categories: list[PeripheralCategory] = Field(default_factory=list)
    motivation: str = ""
    leverage: str = ""  # what leverage they hold, if any
    feedback_mechanism: str = ""  # how their actions feed back into the primary dynamics
    threshold_for_action: str = ""  # what would cause them to shift categories or act
    influence_estimate: str = "low"  # low | medium | high


class ActorAnalysis(BaseModel):
    actors_involved: bool = False
    actors: list[Actor] = Field(default_factory=list)
    level_2_3_present: bool = False
    peripheral_actors: list[PeripheralActor] = Field(default_factory=list)
    probability_space_constraints: list[str] = Field(default_factory=list)


class Decomposition(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    reference_class: str = ""
    base_rate: float = 0.5
    base_rate_source: str = ""
    reference_class_instance_count: int | None = None
    actor_analysis: ActorAnalysis = Field(default_factory=ActorAnalysis)


class RunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    provider: str = ""
    search_persona: str = SearchPersona.NEWS.value
    search_queries: list[str] = Field(default_factory=list)
    sources_cited: list[str] = Field(default_factory=list)
    search_quality_score: int = 0
    base_rate_used: float = 0.5
    inside_view_adjustment: float = 0.0
    adjustment_reasoning: str = ""
    pre_mortem_scenario: str = ""
    pre_mortem_changed_estimate: bool = False
    pre_mortem_delta: float = 0.0
    actor_motivation_flags: list[str] = Field(default_factory=list)
    probability: float = 0.5
    confidence_self_score: int = 5
    reasoning_chain: str = ""
    parse_failed: bool = False
    falsification_anchors: list[str] = Field(default_factory=list)


class Aggregation(BaseModel):
    method: str = "single_run"
    raw_aggregate: float = 0.5
    stdev: float = 0.0
    inter_model_agreement: float = 1.0
    effective_n: int = 1
    deliberation_triggered: bool = False
    deliberation_result: dict | None = None
    sources_clustering_score: float = 0.0   # 0.0=no clustering, 1.0=all runs cite same apex domain
    clustered_domains: list[str] = Field(default_factory=list)  # apex domains with cross-run clustering


class CalibrationData(BaseModel):
    platt_params_used: dict | None = None  # {"a": X, "b": Y}
    calibrated_probability: float = 0.5
    confidence_interval: list[float] = Field(default_factory=lambda: [0.0, 1.0])


class PlatformSignals(BaseModel):
    prices_at_prediction_time: dict[str, float | None] = Field(default_factory=dict)
    cross_platform_divergence: float | None = None
    anomaly_detected: bool = False
    anomaly_details: str | None = None


class HumanReview(BaseModel):
    reviewed: bool = False
    human_adjustment: float | None = None
    human_reasoning: str | None = None
    flags_raised: list[str] = Field(default_factory=list)
    actor_analysis_consulted: bool = False


class Submission(BaseModel):
    probability: float = 0.5
    submitted_at: str = ""
    platform: str = ""


class Resolution(BaseModel):
    outcome: float | None = None  # 0|1 for binary, value for numeric
    resolved_at: str = ""
    brier_score: float | None = None
    log_score: float | None = None
    error_class_auto: list[str] = Field(default_factory=list)
    error_class_manual: list[str] = Field(default_factory=list)
    structured_lessons: list[str] = Field(default_factory=list)


class VerificationEntry(BaseModel):
    provider: str = ""
    model: str = ""
    assessment: Literal["agree", "disagree", "partial", "uncertain"] = "uncertain"
    confidence: Literal["high", "medium", "low"] = "medium"
    reasoning: str = ""
    counter_evidence: str = ""
    status: str = "success"
    error_class: str = ""


class ChallengeEntry(BaseModel):
    provider: str = ""
    model: str = ""
    counter_argument: str = ""
    logical_gaps: str = ""
    evidence_needed: str = ""
    vulnerability: Literal["high", "medium", "low"] = "medium"
    status: str = "success"


class StageVerification(BaseModel):
    stage: str = ""
    finding_summary: str = ""
    verifications: list[VerificationEntry] = Field(default_factory=list)
    challenges: list[ChallengeEntry] = Field(default_factory=list)
    agreement: str = ""  # unanimous | partial | none | no_data


class VerificationReport(BaseModel):
    enabled: bool = False
    providers_available: list[str] = Field(default_factory=list)
    stages: list[StageVerification] = Field(default_factory=list)


class CostTracking(BaseModel):
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    latency_seconds: float = 0.0


class PredictionRecord(BaseModel):
    """Full prediction record matching the registry schema."""
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str = ""
    platform: str = Platform.METACULUS.value
    question_text: str = ""
    question_type: str = QuestionType.BINARY.value
    domain: str = ""
    horizon_days: int | None = None
    close_date: str = ""
    tags: list[str] = Field(default_factory=list)
    pipeline_warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    decomposition: Decomposition = Field(default_factory=Decomposition)
    runs: list[RunResult] = Field(default_factory=list)
    aggregation: Aggregation = Field(default_factory=Aggregation)
    calibration: CalibrationData = Field(default_factory=CalibrationData)
    platform_signals: PlatformSignals = Field(default_factory=PlatformSignals)
    human_review: HumanReview = Field(default_factory=HumanReview)
    submitted: Submission | None = None
    resolution: Resolution | None = None
    verification: VerificationReport = Field(default_factory=VerificationReport)
    cost: CostTracking = Field(default_factory=CostTracking)


class PlatformQuestion(BaseModel):
    """Normalized question from any platform."""
    question_id: str
    platform: Platform
    title: str
    description: str = ""
    question_type: QuestionType = QuestionType.BINARY
    resolution_criteria: str = ""
    fine_print: str = ""
    created_at: str = ""
    close_date: str = ""
    resolve_date: str = ""
    community_prediction: float | None = None
    url: str = ""
    tags: list[str] = Field(default_factory=list)
