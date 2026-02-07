"""Tests for harness and executor."""

import pytest
from llm_eval import Task, Suite, ExactMatchGrader
from llm_eval.harness.base import SimpleHarness
from llm_eval.harness.executor import Executor


def simple_agent(input_data):
    """Simple test agent."""
    return input_data.get("expected_output", "default")


def test_simple_harness():
    """Test simple harness execution."""
    task = Task(
        id="test",
        description="Test",
        input_data={"expected_output": "hello"},
        graders=[ExactMatchGrader("hello")],
    )

    harness = SimpleHarness(agent_fn=simple_agent)
    executor = Executor(harness=harness, verbose=False)

    results = executor.run_task(task, num_trials=1)

    assert len(results) == 1
    assert results[0].passed


def test_multiple_trials():
    """Test running multiple trials."""
    task = Task(
        id="test",
        description="Test",
        input_data={"expected_output": "hello"},
        graders=[ExactMatchGrader("hello")],
    )

    harness = SimpleHarness(agent_fn=simple_agent)
    executor = Executor(harness=harness, verbose=False)

    results = executor.run_task(task, num_trials=5)

    assert len(results) == 5
    assert all(r.passed for r in results)


def test_suite_execution():
    """Test executing a full suite."""
    tasks = [
        Task(
            id="task1",
            description="Task 1",
            input_data={"expected_output": "output1"},
            graders=[ExactMatchGrader("output1")],
        ),
        Task(
            id="task2",
            description="Task 2",
            input_data={"expected_output": "output2"},
            graders=[ExactMatchGrader("output2")],
        ),
    ]

    suite = Suite(name="test_suite", tasks=tasks)

    harness = SimpleHarness(agent_fn=simple_agent)
    executor = Executor(harness=harness, verbose=False)

    results = executor.run_suite(suite, num_trials=2)

    assert results.num_tasks == 2
    assert results.num_trials == 4  # 2 tasks * 2 trials
    assert results.pass_rate == 1.0


def test_failed_trial():
    """Test handling of failed trials."""
    def failing_agent(input_data):
        raise ValueError("Agent failed")

    task = Task(
        id="test",
        description="Test",
        input_data={},
        graders=[ExactMatchGrader("hello")],
    )

    harness = SimpleHarness(agent_fn=failing_agent)
    executor = Executor(harness=harness, verbose=False)

    results = executor.run_task(task, num_trials=1)

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].error is not None
