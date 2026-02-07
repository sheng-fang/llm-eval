"""Trial execution and result management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from llm_eval.core.transcript import Transcript
from llm_eval.graders.base import GraderResult


class TrialStatus(Enum):
    """Status of a trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrialResult:
    """
    Result of a single trial.

    Attributes:
        task_id: ID of the task that was run
        trial_number: Trial number (for multiple trials of same task)
        status: Status of the trial
        output: The agent's output
        outcome: Final state in the environment
        transcript: Complete interaction history
        grader_results: Results from all graders
        start_time: When the trial started
        end_time: When the trial ended
        error: Error message if trial failed
        metadata: Additional metadata
    """

    task_id: str
    trial_number: int
    status: TrialStatus
    output: Any = None
    outcome: Optional[dict[str, Any]] = None
    transcript: Optional[Transcript] = None
    grader_results: list[GraderResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get trial duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def passed(self) -> bool:
        """Check if trial passed (all graders passed)."""
        if self.status != TrialStatus.COMPLETED:
            return False
        return all(result.passed for result in self.grader_results)

    @property
    def score(self) -> float:
        """Get average score across all graders."""
        if not self.grader_results:
            return 0.0
        return sum(result.score for result in self.grader_results) / len(self.grader_results)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "task_id": self.task_id,
            "trial_number": self.trial_number,
            "status": self.status.value,
            "passed": self.passed,
            "score": self.score,
            "output": self.output,
            "outcome": self.outcome,
            "grader_results": [gr.to_dict() for gr in self.grader_results],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error": self.error,
            "metadata": self.metadata,
        }


class Trial:
    """
    Manages the execution of a single trial.

    A trial represents one attempt at completing a task. Multiple trials
    are typically run for each task to handle non-deterministic behavior.
    """

    def __init__(self, task_id: str, trial_number: int) -> None:
        """
        Initialize a trial.

        Args:
            task_id: ID of the task to run
            trial_number: Trial number
        """
        self.task_id = task_id
        self.trial_number = trial_number
        self.status = TrialStatus.PENDING
        self.transcript = Transcript()
        self.output: Any = None
        self.outcome: Optional[dict[str, Any]] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self.metadata: dict[str, Any] = {}

    def start(self) -> None:
        """Mark trial as started."""
        self.status = TrialStatus.RUNNING
        self.start_time = datetime.now()
        self.transcript.start()

    def complete(self, output: Any, outcome: Optional[dict[str, Any]] = None) -> None:
        """
        Mark trial as completed.

        Args:
            output: The agent's output
            outcome: Final state in the environment
        """
        self.status = TrialStatus.COMPLETED
        self.end_time = datetime.now()
        self.transcript.end()
        self.output = output
        self.outcome = outcome

    def fail(self, error: str) -> None:
        """
        Mark trial as failed.

        Args:
            error: Error message
        """
        self.status = TrialStatus.FAILED
        self.end_time = datetime.now()
        self.transcript.end()
        self.error = error

    def to_result(self, grader_results: list[GraderResult]) -> TrialResult:
        """
        Convert trial to result.

        Args:
            grader_results: Results from grading this trial

        Returns:
            TrialResult instance
        """
        return TrialResult(
            task_id=self.task_id,
            trial_number=self.trial_number,
            status=self.status,
            output=self.output,
            outcome=self.outcome,
            transcript=self.transcript,
            grader_results=grader_results,
            start_time=self.start_time,
            end_time=self.end_time,
            error=self.error,
            metadata=self.metadata,
        )
