"""
LLM Eval - A modular evaluation framework for LLM applications and AI agents.
"""

from llm_eval.core.suite import Suite
from llm_eval.core.task import Task
from llm_eval.core.trial import Trial, TrialResult
from llm_eval.graders.base import Grader, GraderResult
from llm_eval.graders.code_grader import (
    ContainsGrader,
    ExactMatchGrader,
    JsonSchemaGrader,
    PythonAssertionGrader,
    RegexGrader,
)
from llm_eval.graders.specialized_graders import (
    FormatGrader,
    LengthRangeGrader,
    NumericGrader,
    SemanticSimilarityGrader,
    SimilarityGrader,
)

__version__ = "0.1.0"

__all__ = [
    "Task",
    "Suite",
    "Trial",
    "TrialResult",
    "Grader",
    "GraderResult",
    "ExactMatchGrader",
    "RegexGrader",
    "JsonSchemaGrader",
    "PythonAssertionGrader",
    "ContainsGrader",
    "NumericGrader",
    "SimilarityGrader",
    "SemanticSimilarityGrader",
    "LengthRangeGrader",
    "FormatGrader",
]
