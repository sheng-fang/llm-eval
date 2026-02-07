"""Task definition and management."""

from dataclasses import dataclass, field
from typing import Any, Optional

from llm_eval.graders.base import Grader


@dataclass
class Task:
    """
    Represents a single evaluation task.

    A task defines what to test, including inputs, expected outputs,
    and graders to evaluate the agent's performance.

    Attributes:
        id: Unique identifier for the task
        description: Human-readable description of what the task tests
        input_data: Input data for the agent (e.g., prompt, context)
        graders: List of graders to evaluate the output
        reference_solution: Optional known-good solution for validation
        metadata: Additional metadata (difficulty, category, tags, etc.)
    """

    id: str
    description: str
    input_data: dict[str, Any]
    graders: list[Grader] = field(default_factory=list)
    reference_solution: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task configuration."""
        if not self.id:
            raise ValueError("Task ID cannot be empty")
        if not self.description:
            raise ValueError("Task description cannot be empty")
        if not self.graders:
            raise ValueError("Task must have at least one grader")

    @property
    def category(self) -> Optional[str]:
        """Get task category from metadata."""
        return self.metadata.get("category")

    @property
    def difficulty(self) -> Optional[str]:
        """Get task difficulty from metadata."""
        return self.metadata.get("difficulty")

    @property
    def tags(self) -> list[str]:
        """Get task tags from metadata."""
        return self.metadata.get("tags", [])

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "input_data": self.input_data,
            "reference_solution": self.reference_solution,
            "metadata": self.metadata,
            "num_graders": len(self.graders),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], graders: list[Grader]) -> "Task":
        """
        Create a task from dictionary representation.

        Args:
            data: Dictionary containing task data
            graders: List of grader instances to use

        Returns:
            Task instance
        """
        return cls(
            id=data["id"],
            description=data["description"],
            input_data=data["input_data"],
            graders=graders,
            reference_solution=data.get("reference_solution"),
            metadata=data.get("metadata", {}),
        )
