"""Code-based (deterministic) graders."""

import json
import re
from typing import Any, Callable, Dict, Optional, Pattern

from llm_eval.graders.base import Grader, GraderResult


class ExactMatchGrader(Grader):
    """Grades based on exact string matching."""

    def __init__(
        self,
        expected: str,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize exact match grader.
        
        Args:
            expected: Expected output string
            case_sensitive: Whether to match case
            strip_whitespace: Whether to strip leading/trailing whitespace
            name: Optional custom name
        """
        super().__init__(name)
        self.expected = expected
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output against expected string."""
        expected_str = expected if expected is not None else self.expected
        output_str = str(output)

        if self.strip_whitespace:
            output_str = output_str.strip()
            expected_str = expected_str.strip()

        if not self.case_sensitive:
            output_str = output_str.lower()
            expected_str = expected_str.lower()

        passed = output_str == expected_str
        score = 1.0 if passed else 0.0

        feedback = "Exact match" if passed else f"Expected '{expected_str}', got '{output_str}'"

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
        )


class RegexGrader(Grader):
    """Grades based on regex pattern matching."""

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize regex grader.
        
        Args:
            pattern: Regex pattern to match
            flags: Regex flags (e.g., re.IGNORECASE)
            name: Optional custom name
        """
        super().__init__(name)
        self.pattern: Pattern[str] = re.compile(pattern, flags)
        self.pattern_str = pattern

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output against regex pattern."""
        output_str = str(output)
        match = self.pattern.search(output_str)
        passed = match is not None
        score = 1.0 if passed else 0.0

        feedback = (
            f"Pattern '{self.pattern_str}' matched"
            if passed
            else f"Pattern '{self.pattern_str}' not found"
        )

        details = {}
        if match:
            details["match"] = match.group(0)
            details["groups"] = match.groups()

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details=details,
        )


class JsonSchemaGrader(Grader):
    """Grades JSON output against a schema."""

    def __init__(
        self,
        schema: Dict[str, Any],
        strict: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize JSON schema grader.
        
        Args:
            schema: JSON schema to validate against
            strict: Whether to require exact schema match
            name: Optional custom name
        """
        super().__init__(name)
        self.schema = schema
        self.strict = strict

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output against JSON schema."""
        try:
            # Parse output as JSON if it's a string
            if isinstance(output, str):
                output_json = json.loads(output)
            else:
                output_json = output

            # Simple schema validation (for full validation, use jsonschema library)
            errors = self._validate_schema(output_json, self.schema)

            passed = len(errors) == 0
            score = 1.0 if passed else 0.0
            feedback = "Valid JSON schema" if passed else f"Schema errors: {'; '.join(errors)}"

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=feedback,
                details={"errors": errors},
            )

        except json.JSONDecodeError as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Invalid JSON: {str(e)}",
            )

    def _validate_schema(
        self, data: Any, schema: Dict[str, Any], path: str = "root"
    ) -> list[str]:
        """Simple schema validation (basic implementation)."""
        errors = []

        if "type" in schema:
            expected_type = schema["type"]
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }

            if expected_type in type_map:
                if not isinstance(data, type_map[expected_type]):
                    errors.append(f"{path}: expected {expected_type}, got {type(data).__name__}")

        if "properties" in schema and isinstance(data, dict):
            for key, prop_schema in schema["properties"].items():
                if key in data:
                    errors.extend(self._validate_schema(data[key], prop_schema, f"{path}.{key}"))
                elif schema.get("required", []) and key in schema["required"]:
                    errors.append(f"{path}.{key}: required property missing")

        return errors


class PythonAssertionGrader(Grader):
    """Grades using a custom Python assertion function."""

    def __init__(
        self,
        assertion_fn: Callable[[Any], bool],
        error_message: str = "Assertion failed",
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize assertion grader.
        
        Args:
            assertion_fn: Function that takes output and returns True if valid
            error_message: Error message when assertion fails
            name: Optional custom name
        """
        super().__init__(name)
        self.assertion_fn = assertion_fn
        self.error_message = error_message

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output using assertion function."""
        try:
            passed = self.assertion_fn(output)
            score = 1.0 if passed else 0.0
            feedback = "Assertion passed" if passed else self.error_message

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=feedback,
            )

        except Exception as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Assertion error: {str(e)}",
            )


class ContainsGrader(Grader):
    """Grades based on whether output contains expected substring(s)."""

    def __init__(
        self,
        expected_substrings: list[str],
        require_all: bool = True,
        case_sensitive: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize contains grader.
        
        Args:
            expected_substrings: List of substrings to check for
            require_all: If True, all substrings must be present; if False, any one is enough
            case_sensitive: Whether to match case
            name: Optional custom name
        """
        super().__init__(name)
        self.expected_substrings = expected_substrings
        self.require_all = require_all
        self.case_sensitive = case_sensitive

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output based on substring presence."""
        output_str = str(output)

        if not self.case_sensitive:
            output_str = output_str.lower()
            substrings = [s.lower() for s in self.expected_substrings]
        else:
            substrings = self.expected_substrings

        found = [s for s in substrings if s in output_str]
        missing = [s for s in substrings if s not in output_str]

        if self.require_all:
            passed = len(missing) == 0
            feedback = (
                "All substrings found"
                if passed
                else f"Missing substrings: {', '.join(missing)}"
            )
        else:
            passed = len(found) > 0
            feedback = (
                f"Found substring(s): {', '.join(found)}"
                if passed
                else "No expected substrings found"
            )

        score = len(found) / len(substrings) if substrings else 0.0

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details={"found": found, "missing": missing},
        )
