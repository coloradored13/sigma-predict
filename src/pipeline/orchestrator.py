"""Main pipeline orchestrator.

Coordinates the full prediction pipeline:
1. Fetch question from platform
2. Decompose question (sub-questions, actor analysis)
3. Estimate base rate (isolated, before search)
4. Run forecaster(s) with independent search
5. Aggregate estimates
6. Calibrate
7. Present for human review
8. Submit on approval
9. Log to registry
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.config import Config
from src.models import (
    Aggregation,
    CalibrationData,
    CostTracking,
    HumanReview,
    PlatformQuestion,
    PlatformSignals,
    PredictionRecord,
    SearchPersona,
    Submission,
    VerificationReport,
)
from src.pipeline.aggregator import aggregate_runs
from src.pipeline.base_rate import estimate_base_rate
from src.pipeline.calibration import calibrate, load_platt_params
from src.pipeline.decomposer import decompose_question
from src.pipeline.deliberation import deliberate, should_deliberate
from src.pipeline.forecaster import run_forecast
from src.pipeline.llm_router import LLMRouter
from src.pipeline.verification import (
    challenge_forecast,
    should_verify_run,
    verify_actors,
    verify_base_rate,
    verify_forecast_run,
)
from src.platforms.base import PlatformClient
from src.registry.store import RegistryStore
from src.registry.query import RegistryQuery
from src.review.human_review import present_for_review

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main prediction pipeline orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self.router = LLMRouter(config)
        self.store = RegistryStore(config.registry_path)
        self.query = RegistryQuery(self.store)

    def predict(
        self,
        question: PlatformQuestion,
        platform_client: PlatformClient,
        num_runs: int = 1,
        personas: list[SearchPersona] | None = None,
        auto_submit: bool = False,
    ) -> PredictionRecord:
        """Run the full prediction pipeline on a question.

        Args:
            question: The question to predict on
            platform_client: Client for the question's platform
            num_runs: Number of independent estimation runs
            personas: Search personas to use (one per run)
            auto_submit: If True, skip human review (for testing)

        Returns:
            Complete PredictionRecord
        """
        pipeline_start = time.time()
        total_cost = CostTracking()

        if personas is None:
            if num_runs == 1:
                personas = [SearchPersona.NEWS]
            else:
                # Rotate through personas
                all_personas = list(SearchPersona)
                personas = [all_personas[i % len(all_personas)] for i in range(num_runs)]

        logger.info("Starting prediction pipeline for: %s", question.title)

        # Initialize record
        record = PredictionRecord(
            question_id=question.question_id,
            platform=question.platform.value,
            question_text=question.title,
            question_type=question.question_type.value,
            domain=self._infer_domain(question),
            horizon_days=self._compute_horizon(question),
        )

        # Set up verification report
        v_config = self.config.verification
        v_enabled = v_config.enabled and self.router.verification_available()
        record.verification = VerificationReport(
            enabled=v_enabled,
            providers_available=self.router.available_providers(),
        )
        if v_config.enabled and not self.router.verification_available():
            logger.warning(
                "Verification enabled in config but no external providers available"
            )

        question_text_full = question.title + "\n" + question.description

        # Step 1: Decompose question
        logger.info("Step 1/7: Decomposing question...")
        decomposition = decompose_question(question, self.config, router=self.router)
        record.decomposition = decomposition

        # Step 1b: Verify actors if enabled and L2/3 present
        if v_enabled and v_config.verify_actors:
            if decomposition.actor_analysis.level_2_3_present:
                logger.info("  Step 1b: Verifying actor analysis...")
                actor_sv = verify_actors(
                    decomposition.actor_analysis,
                    question_text_full,
                    self.router,
                    self.config,
                )
                if actor_sv is not None:
                    record.verification.stages.append(actor_sv)

        # Step 2: Estimate base rate (isolated — BEFORE search)
        logger.info("Step 2/7: Estimating base rate (isolated)...")
        base_rate, ref_class, base_source = estimate_base_rate(
            question_text=question_text_full,
            sub_questions=decomposition.sub_questions,
            reference_class=decomposition.reference_class,
            config=self.config,
            router=self.router,
        )
        record.decomposition.base_rate = base_rate
        record.decomposition.reference_class = ref_class
        record.decomposition.base_rate_source = base_source

        # Step 2b: Verify base rate if enabled
        if v_enabled and v_config.verify_base_rate:
            logger.info("  Step 2b: Verifying base rate...")
            br_sv = verify_base_rate(
                base_rate, ref_class, question_text_full,
                self.router, self.config,
            )
            record.verification.stages.append(br_sv)

        # Step 3: Run forecaster(s)
        logger.info("Step 3/7: Running %d forecast run(s)...", num_runs)
        from src.pipeline.search import SearchModule
        search_module = SearchModule(self.config)

        for i, persona in enumerate(personas[:num_runs]):
            logger.info("  Run %d/%d: model=%s persona=%s",
                       i + 1, num_runs,
                       self.config.primary_model.model_id,
                       persona.value)

            # Independent search for this run
            search_results = search_module.search_with_persona(
                question.title + " " + question.description[:200],
                persona,
            )

            run_result, run_cost = run_forecast(
                question=question,
                decomposition=decomposition,
                base_rate=base_rate,
                base_rate_source=base_source,
                search_results=search_results,
                persona=persona,
                config=self.config,
                router=self.router,
            )
            record.runs.append(run_result)
            total_cost = _add_costs(total_cost, run_cost)

        # Step 3b: Verify forecast runs (outliers or all)
        if v_enabled and v_config.verify_forecasts != "none":
            for run in record.runs:
                if should_verify_run(run, record.runs, self.config):
                    logger.info("  Step 3b: Verifying forecast run %s...", run.run_id[:8])
                    run_sv = verify_forecast_run(
                        run, question_text_full,
                        self.router, self.config,
                    )
                    record.verification.stages.append(run_sv)

        # Step 3c: Challenge forecast via pre-mortem if enabled
        if v_enabled and v_config.challenge_pre_mortem and record.runs:
            logger.info("  Step 3c: Challenging forecast (pre-mortem)...")
            # Challenge the last run as representative of the aggregate
            challenge_entry = challenge_forecast(
                record.runs[-1], question_text_full,
                self.router, self.config,
            )
            if challenge_entry is not None:
                # Attach challenge to a StageVerification
                from src.models import StageVerification
                challenge_sv = StageVerification(
                    stage="challenge_pre_mortem",
                    finding_summary=f"Challenge of forecast P={record.runs[-1].probability:.3f}",
                    challenges=[challenge_entry],
                    agreement="no_data",  # challenges don't have agreement
                )
                record.verification.stages.append(challenge_sv)

        # Step 4: Aggregate
        logger.info("Step 4/7: Aggregating estimates...")
        aggregation = aggregate_runs(
            record.runs,
            method=self.config.aggregation.method,
            trim_fraction=self.config.aggregation.trim_fraction,
            extremization_factor=self.config.aggregation.extremization_factor,
        )
        record.aggregation = aggregation

        # Check for deliberation (Phase 3)
        if should_deliberate(aggregation, self.config) and num_runs > 1:
            logger.info("Deliberation triggered (stdev=%.3f)", aggregation.stdev)
            record.runs, record.aggregation = deliberate(
                record.runs, aggregation, self.config
            )
            record.aggregation.deliberation_triggered = True

        # Step 5: Calibrate
        logger.info("Step 5/7: Calibrating...")
        n_resolved = self.query.count_resolved()
        platt_params = load_platt_params(self.config)

        cal_prob, platt_dict = calibrate(
            probability=aggregation.raw_aggregate,
            config=self.config,
            platt_params=platt_params,
            n_resolved=n_resolved,
        )

        # Compute confidence interval from stdev
        stdev = aggregation.stdev
        ci_low = max(0.01, cal_prob - 1.96 * stdev) if stdev > 0 else max(0.01, cal_prob - 0.05)
        ci_high = min(0.99, cal_prob + 1.96 * stdev) if stdev > 0 else min(0.99, cal_prob + 0.05)

        record.calibration = CalibrationData(
            platt_params_used=platt_dict,
            calibrated_probability=cal_prob,
            confidence_interval=[round(ci_low, 4), round(ci_high, 4)],
        )

        # Collect platform signals
        record.platform_signals = self._collect_platform_signals(question, platform_client)

        # Step 6: Human review
        logger.info("Step 6/7: Presenting for human review...")
        total_cost.latency_seconds = time.time() - pipeline_start
        record.cost = total_cost

        if auto_submit:
            record.human_review = HumanReview(reviewed=False)
            final_prob = cal_prob
        else:
            review = present_for_review(record, question)
            record.human_review = review

            if review.human_reasoning and review.human_reasoning.startswith("REJECTED"):
                logger.info("Prediction rejected by human reviewer")
                self.store.save(record)
                return record

            if review.human_adjustment is not None:
                final_prob = cal_prob + review.human_adjustment
                final_prob = max(0.01, min(0.99, final_prob))
            else:
                final_prob = cal_prob

        # Submit
        logger.info("Submitting prediction: %.3f to %s", final_prob, question.platform.value)
        try:
            success = platform_client.submit_prediction(
                question.question_id, final_prob
            )
            if success:
                record.submitted = Submission(
                    probability=final_prob,
                    submitted_at=datetime.now(timezone.utc).isoformat(),
                    platform=question.platform.value,
                )
                logger.info("Prediction submitted successfully")
            else:
                logger.warning("Prediction submission failed")
        except NotImplementedError:
            logger.info("Submission not implemented for %s — logging only", question.platform.value)
            record.submitted = Submission(
                probability=final_prob,
                submitted_at=datetime.now(timezone.utc).isoformat(),
                platform=question.platform.value + "_logged_only",
            )

        # Save to registry
        self.store.save(record)
        logger.info(
            "Prediction logged: id=%s prob=%.3f cost=$%.4f",
            record.prediction_id, final_prob, total_cost.total_cost_usd,
        )

        return record

    def _infer_domain(self, question: PlatformQuestion) -> str:
        """Infer the domain from question tags or content."""
        domain_keywords = {
            "geopolitics": ["war", "conflict", "treaty", "diplomacy", "sanctions", "military"],
            "technology": ["ai", "tech", "software", "computing", "internet", "crypto"],
            "science": ["climate", "vaccine", "research", "study", "discovery", "space"],
            "economics": ["gdp", "inflation", "market", "trade", "recession", "employment"],
            "politics": ["election", "vote", "president", "congress", "legislation", "policy"],
            "health": ["disease", "pandemic", "health", "medical", "drug", "fda"],
            "sports": ["game", "championship", "tournament", "team", "player", "season"],
        }

        text = (question.title + " " + " ".join(question.tags)).lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return domain
        return "general"

    def _compute_horizon(self, question: PlatformQuestion) -> int | None:
        """Compute days until resolution."""
        if not question.resolve_date:
            return None
        try:
            resolve = datetime.fromisoformat(question.resolve_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (resolve - now).days)
        except (ValueError, TypeError):
            return None

    def _collect_platform_signals(
        self,
        question: PlatformQuestion,
        platform_client: PlatformClient,
    ) -> PlatformSignals:
        """Collect current platform prices for comparison."""
        signals = PlatformSignals()

        try:
            community = platform_client.get_community_prediction(question.question_id)
            if community is not None:
                platform_key = f"{question.platform.value}_community"
                if platform_key in signals.prices_at_prediction_time:
                    signals.prices_at_prediction_time[platform_key] = community
                else:
                    signals.prices_at_prediction_time[question.platform.value] = community
        except Exception as e:
            logger.debug("Failed to get community prediction: %s", e)

        return signals


def _add_costs(a: CostTracking, b: CostTracking) -> CostTracking:
    """Add two cost tracking objects together."""
    return CostTracking(
        total_tokens_in=a.total_tokens_in + b.total_tokens_in,
        total_tokens_out=a.total_tokens_out + b.total_tokens_out,
        total_cost_usd=a.total_cost_usd + b.total_cost_usd,
        latency_seconds=a.latency_seconds + b.latency_seconds,
    )
