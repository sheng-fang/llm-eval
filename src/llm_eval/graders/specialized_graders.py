"""Additional specialized graders for various use cases."""

import re
from typing import Any, Optional, List, Callable
from difflib import SequenceMatcher

from llm_eval.graders.base import Grader, GraderResult


class NumericGrader(Grader):
    """
    Grades numeric outputs with tolerance.
    
    Useful for evaluating mathematical computations, measurements, etc.
    """

    def __init__(
        self,
        expected: float,
        tolerance: float = 0.01,
        relative: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize numeric grader.
        
        Args:
            expected: Expected numeric value
            tolerance: Acceptable deviation (absolute or relative)
            relative: If True, tolerance is relative (percentage); if False, absolute
            name: Optional custom name
        """
        super().__init__(name)
        self.expected = expected
        self.tolerance = tolerance
        self.relative = relative

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade numeric output."""
        expected_val = expected if expected is not None else self.expected

        try:
            # Extract number from output
            actual = self._extract_number(output)

            if self.relative:
                # Relative tolerance (percentage)
                if expected_val == 0:
                    passed = abs(actual) <= self.tolerance
                    error = abs(actual)
                else:
                    error = abs((actual - expected_val) / expected_val)
                    passed = error <= self.tolerance
            else:
                # Absolute tolerance
                error = abs(actual - expected_val)
                passed = error <= self.tolerance

            # Score based on error (closer = higher score)
            if passed:
                score = 1.0 - (error / self.tolerance) * 0.5  # 0.5 to 1.0 range
            else:
                score = max(0.0, 1.0 - error)

            feedback = (
                f"Expected {expected_val}, got {actual} "
                f"(error: {error:.4f}, tolerance: {self.tolerance})"
            )

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=feedback,
                details={"expected": expected_val, "actual": actual, "error": error},
            )

        except (ValueError, TypeError) as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Could not parse numeric value: {str(e)}",
            )

    def _extract_number(self, output: Any) -> float:
        """Extract numeric value from output."""
        if isinstance(output, (int, float)):
            return float(output)

        # Try to parse from string
        text = str(output)

        # Look for numbers (including scientific notation)
        pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        matches = re.findall(pattern, text)

        if not matches:
            raise ValueError(f"No numeric value found in: {text}")

        # Return the first number found
        return float(matches[0])


class SimilarityGrader(Grader):
    """
    Grades based on string similarity using various algorithms.
    
    Useful for fuzzy matching when exact match is too strict.
    """

    def __init__(
        self,
        expected: str,
        threshold: float = 0.8,
        algorithm: str = "sequence_matcher",
        case_sensitive: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize similarity grader.
        
        Args:
            expected: Expected output string
            threshold: Minimum similarity score to pass (0.0 to 1.0)
            algorithm: Similarity algorithm ('sequence_matcher', 'jaccard', 'levenshtein')
            case_sensitive: Whether to match case
            name: Optional custom name
        """
        super().__init__(name)
        self.expected = expected
        self.threshold = threshold
        self.algorithm = algorithm
        self.case_sensitive = case_sensitive

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade based on similarity."""
        expected_str = expected if expected is not None else self.expected
        output_str = str(output)

        if not self.case_sensitive:
            expected_str = expected_str.lower()
            output_str = output_str.lower()

        # Calculate similarity
        if self.algorithm == "sequence_matcher":
            similarity = SequenceMatcher(None, expected_str, output_str).ratio()
        elif self.algorithm == "jaccard":
            similarity = self._jaccard_similarity(expected_str, output_str)
        elif self.algorithm == "levenshtein":
            similarity = self._levenshtein_similarity(expected_str, output_str)
        else:
            similarity = SequenceMatcher(None, expected_str, output_str).ratio()

        passed = similarity >= self.threshold
        score = similarity

        feedback = (
            f"Similarity: {similarity:.2%} "
            f"({'✓' if passed else '✗'} threshold: {self.threshold:.2%})"
        )

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details={
                "similarity": similarity,
                "algorithm": self.algorithm,
                "threshold": self.threshold,
            },
        )

    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """Calculate Jaccard similarity between two strings."""
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union)

    def _levenshtein_similarity(self, str1: str, str2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        distance = self._levenshtein_distance(str1, str2)
        max_len = max(len(str1), len(str2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(str1) < len(str2):
            return self._levenshtein_distance(str2, str1)

        if len(str2) == 0:
            return len(str1)

        previous_row = range(len(str2) + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class SemanticSimilarityGrader(Grader):
    """
    Grades based on semantic similarity using embeddings.
    
    Requires an embedding function (e.g., from OpenAI, sentence-transformers).
    """

    def __init__(
        self,
        expected: str,
        embedding_fn: Callable[[str], List[float]],
        threshold: float = 0.85,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize semantic similarity grader.
        
        Args:
            expected: Expected output string
            embedding_fn: Function that takes text and returns embedding vector
            threshold: Minimum cosine similarity to pass (0.0 to 1.0)
            name: Optional custom name
        """
        super().__init__(name)
        self.expected = expected
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self._expected_embedding: Optional[List[float]] = None

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade based on semantic similarity."""
        expected_str = expected if expected is not None else self.expected
        output_str = str(output)

        try:
            # Get embeddings
            if expected == self.expected and self._expected_embedding is not None:
                expected_emb = self._expected_embedding
            else:
                expected_emb = self.embedding_fn(expected_str)
                if expected == self.expected:
                    self._expected_embedding = expected_emb

            output_emb = self.embedding_fn(output_str)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(expected_emb, output_emb)

            passed = similarity >= self.threshold
            score = similarity

            feedback = (
                f"Semantic similarity: {similarity:.2%} "
                f"({'✓' if passed else '✗'} threshold: {self.threshold:.2%})"
            )

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=feedback,
                details={"similarity": similarity, "threshold": self.threshold},
            )

        except Exception as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Embedding error: {str(e)}",
            )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class LengthRangeGrader(Grader):
    """
    Grades based on output length being within a range.
    
    Useful for ensuring responses are appropriately concise or detailed.
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "chars",
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize length range grader.
        
        Args:
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            unit: Unit of measurement ('chars', 'words', 'lines')
            name: Optional custom name
        """
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade based on length."""
        text = str(output)

        # Calculate length based on unit
        if self.unit == "chars":
            length = len(text)
        elif self.unit == "words":
            length = len(text.split())
        elif self.unit == "lines":
            length = len(text.splitlines())
        else:
            length = len(text)

        passed = True
        feedback_parts = []

        if self.min_length is not None and length < self.min_length:
            passed = False
            feedback_parts.append(f"Too short (min: {self.min_length} {self.unit})")

        if self.max_length is not None and length > self.max_length:
            passed = False
            feedback_parts.append(f"Too long (max: {self.max_length} {self.unit})")

        if passed:
            feedback = f"Length OK ({length} {self.unit})"
        else:
            feedback = ", ".join(feedback_parts) + f" (actual: {length} {self.unit})"

        # Partial credit based on how close to acceptable range
        if self.min_length and self.max_length:
            if length < self.min_length:
                score = max(0.0, length / self.min_length)
            elif length > self.max_length:
                overage = length - self.max_length
                score = max(0.0, 1.0 - (overage / self.max_length))
            else:
                score = 1.0
        elif self.min_length:
            score = min(1.0, length / self.min_length) if length < self.min_length else 1.0
        elif self.max_length:
            score = 1.0 if length <= self.max_length else max(0.0, 1.0 - (length - self.max_length) / self.max_length)
        else:
            score = 1.0

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details={"length": length, "unit": self.unit},
        )


class FormatGrader(Grader):
    """
    Grades based on output format (JSON, XML, Markdown, etc.).
    
    Useful for ensuring structured outputs.
    """

    def __init__(
        self,
        format_type: str,
        strict: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize format grader.
        
        Args:
            format_type: Expected format ('json', 'xml', 'markdown', 'yaml')
            strict: Whether to enforce strict format validation
            name: Optional custom name
        """
        super().__init__(name)
        self.format_type = format_type.lower()
        self.strict = strict

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade based on format."""
        text = str(output)

        try:
            if self.format_type == "json":
                import json
                json.loads(text)
                passed = True
                feedback = "Valid JSON format"

            elif self.format_type == "xml":
                import xml.etree.ElementTree as ET
                ET.fromstring(text)
                passed = True
                feedback = "Valid XML format"

            elif self.format_type == "yaml":
                try:
                    import yaml
                    yaml.safe_load(text)
                    passed = True
                    feedback = "Valid YAML format"
                except ImportError:
                    return GraderResult(
                        grader_name=self.name,
                        passed=False,
                        score=0.0,
                        feedback="YAML library not installed (pip install pyyaml)",
                    )

            elif self.format_type == "markdown":
                # Basic markdown validation (has headers, lists, etc.)
                has_headers = bool(re.search(r"^#+\s", text, re.MULTILINE))
                has_lists = bool(re.search(r"^[\*\-\+]\s", text, re.MULTILINE))
                passed = has_headers or has_lists
                feedback = "Valid Markdown format" if passed else "No Markdown formatting detected"

            else:
                passed = False
                feedback = f"Unknown format type: {self.format_type}"

            score = 1.0 if passed else 0.0

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=feedback,
                details={"format_type": self.format_type},
            )

        except Exception as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Format validation failed: {str(e)}",
            )
