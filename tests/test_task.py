"""Tests for core task functionality."""

import pytest
from llm_eval.core.task import Task
from llm_eval.graders.code_grader import ExactMatchGrader


def test_task_creation():
    """Test basic task creation."""
    task = Task(
        id="test_task",
        description="Test task",
        input_data={"prompt": "test"},
        graders=[ExactMatchGrader("expected")],
    )

    assert task.id == "test_task"
    assert task.description == "Test task"
    assert len(task.graders) == 1


def test_task_requires_id():
    """Test that task requires an ID."""
    with pytest.raises(ValueError):
        Task(
            id="",
            description="Test",
            input_data={},
            graders=[ExactMatchGrader("test")],
        )


def test_task_requires_graders():
    """Test that task requires at least one grader."""
    with pytest.raises(ValueError):
        Task(
            id="test",
            description="Test",
            input_data={},
            graders=[],
        )


def test_task_metadata():
    """Test task metadata access."""
    task = Task(
        id="test",
        description="Test",
        input_data={},
        graders=[ExactMatchGrader("test")],
        metadata={"category": "math", "difficulty": "easy", "tags": ["arithmetic"]},
    )

    assert task.category == "math"
    assert task.difficulty == "easy"
    assert "arithmetic" in task.tags


def test_task_to_dict():
    """Test task serialization."""
    task = Task(
        id="test",
        description="Test task",
        input_data={"key": "value"},
        graders=[ExactMatchGrader("test")],
    )

    task_dict = task.to_dict()

    assert task_dict["id"] == "test"
    assert task_dict["description"] == "Test task"
    assert task_dict["input_data"] == {"key": "value"}
    assert task_dict["num_graders"] == 1
