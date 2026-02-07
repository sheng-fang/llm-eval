"""Base grader interface and result classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GraderResult:
    """
    Result from a grader.

    Attributes:
        grader_name: Name of the grader
        passed: Whether the grading passed
        score: Numerical score (0.0 to 1.0)
        feedback: Human-readable feedback
        details: Additional details about the grading
    """

    grader_name: str
    passed: bool
    score: float
    feedback: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate grader result."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "grader_name": self.grader_name,
            "passed": self.passed,
            "score": self.score,
            "feedback": self.feedback,
            "details": self.details,
        }


class Grader(ABC):
    """
    Abstract base class for all graders.

    Graders evaluate agent outputs and return a GraderResult.
    They can be deterministic (code-based) or non-deterministic (model-based).
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initialize grader.

        Args:
            name: Optional custom name for the grader
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """
        Grade an output.

        Args:
            output: The agent's output to grade
            expected: Expected output (if applicable)
            transcript: Optional transcript for context
            **kwargs: Additional grading parameters

        Returns:
            GraderResult with score and feedback
        """
        pass

    async def async_grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """
        Async version of grade. Default implementation calls sync grade.

        Override this method for graders that can benefit from async execution
        (e.g., LLM-based graders making API calls).

        Args:
            output: The agent's output to grade
            expected: Expected output (if applicable)
            transcript: Optional transcript for context
            **kwargs: Additional grading parameters

        Returns:
            GraderResult with score and feedback
        """
        # Default: call sync version
        return self.grade(output, expected, transcript, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class CompositeGrader(Grader):
    """
    Combines multiple graders with a composition strategy.

    Supports AND (all must pass), OR (any must pass), and WEIGHTED combinations.
    """

    def __init__(
        self,
        graders: list[Grader],
        strategy: str = "and",
        weights: Optional[list[float]] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize composite grader.

        Args:
            graders: List of graders to combine
            strategy: Combination strategy ('and', 'or', 'weighted')
            weights: Weights for weighted strategy (must sum to 1.0)
            name: Optional custom name
        """
        super().__init__(name)
        self.graders = graders
        self.strategy = strategy.lower()

        if self.strategy not in ["and", "or", "weighted"]:
            raise ValueError(f"Invalid strategy: {strategy}")

        if self.strategy == "weighted":
            if weights is None:
                # Equal weights
                weights = [1.0 / len(graders)] * len(graders)
            if len(weights) != len(graders):
                raise ValueError("Number of weights must match number of graders")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")

        self.weights = weights

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade using composite strategy."""
        results = [grader.grade(output, expected, transcript, **kwargs) for grader in self.graders]

        if self.strategy == "and":
            passed = all(r.passed for r in results)
            score = min(r.score for r in results)
            feedback = " AND ".join(r.feedback for r in results if r.feedback)

        elif self.strategy == "or":
            passed = any(r.passed for r in results)
            score = max(r.score for r in results)
            feedback = " OR ".join(r.feedback for r in results if r.feedback)

        else:  # weighted
            assert self.weights is not None
            passed = sum(r.score * w for r, w in zip(results, self.weights)) >= 0.5
            score = sum(r.score * w for r, w in zip(results, self.weights))
            feedback = "; ".join(
                f"{r.grader_name} ({w:.1%}): {r.feedback}"
                for r, w in zip(results, self.weights)
                if r.feedback
            )

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details={"individual_results": [r.to_dict() for r in results]},
        )
