"""
LLM Eval - A modular evaluation framework for LLM applications and AI agents.
"""

from llm_eval.core.task import Task
from llm_eval.core.suite import Suite
from llm_eval.core.trial import Trial, TrialResult
from llm_eval.graders.base import Grader, GraderResult
from llm_eval.graders.code_grader import (
    ExactMatchGrader,
    RegexGrader,
    JsonSchemaGrader,
    PythonAssertionGrader,
    ContainsGrader,
)
from llm_eval.graders.specialized_graders import (
    NumericGrader,
    SimilarityGrader,
    SemanticSimilarityGrader,
    LengthRangeGrader,
    FormatGrader,
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
