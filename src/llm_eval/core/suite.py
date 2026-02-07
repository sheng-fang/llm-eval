"""Evaluation suite management."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from llm_eval.core.task import Task


@dataclass
class Suite:
    """
    A collection of related evaluation tasks.

    Suites group tasks that measure specific capabilities or behaviors.
    For example, a customer support suite might test refunds, cancellations,
    and escalations.

    Attributes:
        name: Suite name
        tasks: List of tasks in the suite
        description: Optional description
        metadata: Additional metadata
    """

    name: str
    tasks: list[Task] = field(default_factory=list)
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate suite configuration."""
        if not self.name:
            raise ValueError("Suite name cannot be empty")

    def add_task(self, task: Task) -> None:
        """
        Add a task to the suite.

        Args:
            task: Task to add
        """
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: Task ID to find

        Returns:
            Task if found, None otherwise
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def filter_by_category(self, category: str) -> list[Task]:
        """
        Filter tasks by category.

        Args:
            category: Category to filter by

        Returns:
            List of tasks in the category
        """
        return [task for task in self.tasks if task.category == category]

    def filter_by_tag(self, tag: str) -> list[Task]:
        """
        Filter tasks by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of tasks with the tag
        """
        return [task for task in self.tasks if tag in task.tags]

    @property
    def num_tasks(self) -> int:
        """Get number of tasks in suite."""
        return len(self.tasks)

    @property
    def categories(self) -> list[str]:
        """Get all unique categories in the suite."""
        cats = set()
        for task in self.tasks:
            if task.category:
                cats.add(task.category)
        return sorted(cats)

    def to_dict(self) -> dict[str, Any]:
        """Convert suite to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "num_tasks": self.num_tasks,
            "categories": self.categories,
            "tasks": [task.to_dict() for task in self.tasks],
            "metadata": self.metadata,
        }

    def save(self, filepath: str) -> None:
        """
        Save suite metadata to JSON file.

        Note: This saves task metadata but not grader instances.
        To fully reconstruct a suite, you'll need to recreate graders.

        Args:
            filepath: Path to save the suite
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Suite":
        """
        Load suite metadata from JSON file.

        Note: This loads task metadata but not grader instances.
        You'll need to add graders separately.

        Args:
            filepath: Path to the suite file

        Returns:
            Suite instance (without graders)
        """
        with open(filepath) as f:
            data = json.load(f)

        suite = cls(
            name=data["name"],
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )

        # Note: Tasks loaded without graders - need to be added separately
        for task_data in data["tasks"]:
            task = Task.from_dict(task_data, graders=[])
            suite.add_task(task)

        return suite
