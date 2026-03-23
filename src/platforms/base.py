"""Abstract base class for prediction platform clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Platform, PlatformQuestion, Resolution


class PlatformClient(ABC):
    """Base interface for all platform clients.

    Each platform client must implement methods to fetch questions,
    submit predictions, and retrieve resolutions and community signals.
    """

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The platform this client connects to."""
        ...

    @abstractmethod
    def get_questions(
        self,
        limit: int = 20,
        offset: int = 0,
        tournament_id: int | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
    ) -> list[PlatformQuestion]:
        """Fetch a list of questions from the platform.

        Args:
            limit: Maximum number of questions to return.
            offset: Pagination offset.
            tournament_id: Optional tournament/event filter.
            tags: Optional tag filter.
            search: Optional search query string.

        Returns:
            List of normalized PlatformQuestion objects.
        """
        ...

    @abstractmethod
    def get_question(self, question_id: str) -> PlatformQuestion:
        """Fetch a single question by its platform-specific ID.

        Args:
            question_id: The platform's identifier for the question.

        Returns:
            A normalized PlatformQuestion.

        Raises:
            ValueError: If the question is not found.
        """
        ...

    @abstractmethod
    def submit_prediction(self, question_id: str, probability: float) -> bool:
        """Submit a probability prediction for a question.

        Args:
            question_id: The platform's identifier for the question.
            probability: Predicted probability (0.0 to 1.0).

        Returns:
            True if submission succeeded, False otherwise.
        """
        ...

    @abstractmethod
    def get_resolution(self, question_id: str) -> Resolution | None:
        """Get the resolution for a question, if resolved.

        Args:
            question_id: The platform's identifier for the question.

        Returns:
            A Resolution object if resolved, None otherwise.
        """
        ...

    @abstractmethod
    def get_community_prediction(self, question_id: str) -> float | None:
        """Get the current community/market prediction for a question.

        Args:
            question_id: The platform's identifier for the question.

        Returns:
            Community probability (0.0 to 1.0) or None if unavailable.
        """
        ...
