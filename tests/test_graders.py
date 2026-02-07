"""Tests for grader functionality."""

import pytest
from llm_eval.graders.base import GraderResult, CompositeGrader
from llm_eval.graders.code_grader import (
    ExactMatchGrader,
    RegexGrader,
    ContainsGrader,
    PythonAssertionGrader,
)


def test_exact_match_grader():
    """Test exact match grader."""
    grader = ExactMatchGrader("hello")

    result = grader.grade("hello")
    assert result.passed
    assert result.score == 1.0

    result = grader.grade("Hello")
    assert not result.passed


def test_exact_match_case_insensitive():
    """Test case-insensitive exact match."""
    grader = ExactMatchGrader("hello", case_sensitive=False)

    result = grader.grade("Hello")
    assert result.passed
    assert result.score == 1.0


def test_regex_grader():
    """Test regex grader."""
    grader = RegexGrader(r"\d{3}-\d{4}")

    result = grader.grade("Call me at 555-1234")
    assert result.passed
    assert "555-1234" in result.details["match"]

    result = grader.grade("No phone number here")
    assert not result.passed


def test_contains_grader():
    """Test contains grader."""
    grader = ContainsGrader(["apple", "banana"], require_all=True)

    result = grader.grade("I like apple and banana")
    assert result.passed

    result = grader.grade("I like apple")
    assert not result.passed


def test_contains_grader_any():
    """Test contains grader with any strategy."""
    grader = ContainsGrader(["apple", "banana"], require_all=False)

    result = grader.grade("I like apple")
    assert result.passed

    result = grader.grade("I like orange")
    assert not result.passed


def test_python_assertion_grader():
    """Test Python assertion grader."""
    grader = PythonAssertionGrader(
        assertion_fn=lambda x: int(x) > 10,
        error_message="Value must be greater than 10",
    )

    result = grader.grade("15")
    assert result.passed

    result = grader.grade("5")
    assert not result.passed


def test_composite_grader_and():
    """Test composite grader with AND strategy."""
    grader = CompositeGrader(
        graders=[
            ExactMatchGrader("hello"),
            ContainsGrader(["hello"]),
        ],
        strategy="and",
    )

    result = grader.grade("hello")
    assert result.passed

    result = grader.grade("hello world")
    assert not result.passed  # Fails exact match


def test_composite_grader_or():
    """Test composite grader with OR strategy."""
    grader = CompositeGrader(
        graders=[
            ExactMatchGrader("hello"),
            ExactMatchGrader("hi"),
        ],
        strategy="or",
    )

    result = grader.grade("hello")
    assert result.passed

    result = grader.grade("hi")
    assert result.passed

    result = grader.grade("hey")
    assert not result.passed


def test_grader_result_validation():
    """Test grader result validation."""
    # Valid score
    result = GraderResult(
        grader_name="test",
        passed=True,
        score=0.5,
    )
    assert result.score == 0.5

    # Invalid score
    with pytest.raises(ValueError):
        GraderResult(
            grader_name="test",
            passed=True,
            score=1.5,  # Out of range
        )
