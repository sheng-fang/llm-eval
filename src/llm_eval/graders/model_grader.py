"""Model-based (LLM-as-judge) graders with async support."""

from typing import Any, Callable, Dict, List, Optional, Union
import asyncio

from llm_eval.graders.base import Grader, GraderResult


# Type aliases for LLM functions
SyncLLMFunction = Callable[[str], str]
AsyncLLMFunction = Callable[[str], Any]  # Returns awaitable


class RubricGrader(Grader):
    """
    Grades using an LLM with a detailed rubric.
    
    Supports both sync and async LLM functions.
    """

    def __init__(
        self,
        rubric: str,
        llm_fn: Union[SyncLLMFunction, AsyncLLMFunction],
        model: str = "default",
        parse_score_fn: Optional[Callable[[str], float]] = None,
        is_async: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize rubric grader.
        
        Args:
            rubric: Grading rubric or criteria
            llm_fn: Function that takes a prompt and returns LLM response
            model: Model identifier for logging
            parse_score_fn: Optional function to parse score from LLM response
            is_async: Whether llm_fn is async
            name: Optional custom name
        """
        super().__init__(name)
        self.rubric = rubric
        self.llm_fn = llm_fn
        self.model = model
        self.parse_score_fn = parse_score_fn or self._default_parse_score
        self.is_async = is_async

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output using LLM with rubric (sync version)."""
        if not self.is_async:
            # Sync LLM function - call directly
            prompt = self._build_prompt(output, expected)

            try:
                llm_response = self.llm_fn(prompt)
                score = self.parse_score_fn(llm_response)
                passed = score >= 0.5

                return GraderResult(
                    grader_name=self.name,
                    passed=passed,
                    score=score,
                    feedback=llm_response,
                    details={"model": self.model, "prompt": prompt},
                )

            except Exception as e:
                return GraderResult(
                    grader_name=self.name,
                    passed=False,
                    score=0.0,
                    feedback=f"Grading error: {str(e)}",
                )
        
        # Async LLM function - need to run in event loop
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context but can't await - this shouldn't happen
                raise RuntimeError(
                    "Cannot call sync grade() with async LLM in running event loop. "
                    "Use async_grade() instead."
                )
        except RuntimeError:
            # No event loop in this thread - create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.async_grade(output, expected, transcript, **kwargs))
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        
        # Have a loop but it's not running - use it
        return loop.run_until_complete(self.async_grade(output, expected, transcript, **kwargs))

    async def async_grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output using LLM with rubric (async version)."""
        prompt = self._build_prompt(output, expected)

        try:
            if self.is_async:
                llm_response = await self.llm_fn(prompt)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                llm_response = await loop.run_in_executor(None, self.llm_fn, prompt)
            
            score = self.parse_score_fn(llm_response)
            passed = score >= 0.5

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=llm_response,
                details={"model": self.model, "prompt": prompt},
            )

        except Exception as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Grading error: {str(e)}",
            )

    def _build_prompt(self, output: Any, expected: Any = None) -> str:
        """Build grading prompt."""
        prompt = f"""You are an expert evaluator. Grade the following output based on this rubric:

RUBRIC:
{self.rubric}

OUTPUT TO GRADE:
{output}
"""

        if expected is not None:
            prompt += f"\nEXPECTED OUTPUT:\n{expected}\n"

        prompt += """
Please provide:
1. A score from 0.0 to 1.0
2. Detailed feedback explaining your score

Format your response as:
SCORE: <number>
FEEDBACK: <your feedback>
"""

        return prompt

    def _default_parse_score(self, response: str) -> float:
        """Default score parsing from LLM response."""
        import re

        # Look for "SCORE: X.X" pattern
        match = re.search(r"SCORE:\s*([0-9.]+)", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        # Fallback: look for any number between 0 and 1
        match = re.search(r"\b(0\.[0-9]+|1\.0|0|1)\b", response)
        if match:
            return float(match.group(1))

        # Default to 0.5 if can't parse
        return 0.5


class PairwiseGrader(Grader):
    """
    Grades by comparing two outputs using an LLM.
    
    Supports both sync and async LLM functions.
    """

    def __init__(
        self,
        llm_fn: Union[SyncLLMFunction, AsyncLLMFunction],
        criteria: str = "overall quality",
        model: str = "default",
        is_async: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize pairwise grader.
        
        Args:
            llm_fn: Function that takes a prompt and returns LLM response
            criteria: Criteria for comparison
            model: Model identifier for logging
            is_async: Whether llm_fn is async
            name: Optional custom name
        """
        super().__init__(name)
        self.llm_fn = llm_fn
        self.criteria = criteria
        self.model = model
        self.is_async = is_async

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade by comparing output to expected (sync version)."""
        if expected is None:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback="Pairwise grader requires expected output for comparison",
            )

        if not self.is_async:
            # Sync LLM function
            prompt = self._build_prompt(output, expected)

            try:
                llm_response = self.llm_fn(prompt)
                winner, confidence = self._parse_comparison(llm_response)

                passed = winner in ["A", "TIE"]
                score = confidence if winner == "A" else (0.5 if winner == "TIE" else 1.0 - confidence)

                return GraderResult(
                    grader_name=self.name,
                    passed=passed,
                    score=score,
                    feedback=llm_response,
                    details={"model": self.model, "winner": winner, "confidence": confidence},
                )

            except Exception as e:
                return GraderResult(
                    grader_name=self.name,
                    passed=False,
                    score=0.0,
                    feedback=f"Grading error: {str(e)}",
                )

        # Async LLM function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Use async_grade() in async context")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.async_grade(output, expected, transcript, **kwargs))
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        return loop.run_until_complete(self.async_grade(output, expected, transcript, **kwargs))

    async def async_grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade by comparing output to expected (async version)."""
        if expected is None:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback="Pairwise grader requires expected output for comparison",
            )

        prompt = self._build_prompt(output, expected)

        try:
            if self.is_async:
                llm_response = await self.llm_fn(prompt)
            else:
                loop = asyncio.get_event_loop()
                llm_response = await loop.run_in_executor(None, self.llm_fn, prompt)

            winner, confidence = self._parse_comparison(llm_response)

            passed = winner in ["A", "TIE"]
            score = confidence if winner == "A" else (0.5 if winner == "TIE" else 1.0 - confidence)

            return GraderResult(
                grader_name=self.name,
                passed=passed,
                score=score,
                feedback=llm_response,
                details={"model": self.model, "winner": winner, "confidence": confidence},
            )

        except Exception as e:
            return GraderResult(
                grader_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Grading error: {str(e)}",
            )

    def _build_prompt(self, output: Any, expected: Any) -> str:
        """Build comparison prompt."""
        return f"""You are an expert evaluator. Compare these two outputs based on {self.criteria}.

OUTPUT A:
{output}

OUTPUT B (reference):
{expected}

Which output is better? Respond with:
WINNER: A or B or TIE
CONFIDENCE: <0.0 to 1.0>
REASONING: <your reasoning>
"""

    def _parse_comparison(self, response: str) -> tuple[str, float]:
        """Parse winner and confidence from response."""
        import re

        winner = "TIE"
        confidence = 0.5

        winner_match = re.search(r"WINNER:\s*(A|B|TIE)", response, re.IGNORECASE)
        if winner_match:
            winner = winner_match.group(1).upper()

        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))

        return winner, confidence


class MultiJudgeGrader(Grader):
    """
    Uses multiple LLM judges and aggregates their scores.
    
    Supports async execution for parallel judge calls.
    """

    def __init__(
        self,
        base_grader: Grader,
        num_judges: int = 3,
        aggregation: str = "mean",
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize multi-judge grader.
        
        Args:
            base_grader: Base grader to run multiple times
            num_judges: Number of judges (runs)
            aggregation: How to aggregate scores ('mean', 'median', 'max', 'min')
            name: Optional custom name
        """
        super().__init__(name)
        self.base_grader = base_grader
        self.num_judges = num_judges
        self.aggregation = aggregation

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade using multiple judges (sync version)."""
        results = []
        for _ in range(self.num_judges):
            result = self.base_grader.grade(output, expected, transcript, **kwargs)
            results.append(result)

        return self._aggregate_results(results)

    async def async_grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade using multiple judges in parallel (async version)."""
        tasks = [
            self.base_grader.async_grade(output, expected, transcript, **kwargs)
            for _ in range(self.num_judges)
        ]
        results = await asyncio.gather(*tasks)
        return self._aggregate_results(list(results))

    def _aggregate_results(self, results: List[GraderResult]) -> GraderResult:
        """Aggregate multiple grader results."""
        scores = [r.score for r in results]

        if self.aggregation == "mean":
            final_score = sum(scores) / len(scores)
        elif self.aggregation == "median":
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            final_score = sorted_scores[mid]
        elif self.aggregation == "max":
            final_score = max(scores)
        elif self.aggregation == "min":
            final_score = min(scores)
        else:
            final_score = sum(scores) / len(scores)

        passed = final_score >= 0.5
        feedback = f"Multi-judge ({self.num_judges} judges, {self.aggregation}): {final_score:.2f}"

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=final_score,
            feedback=feedback,
            details={
                "individual_results": [r.to_dict() for r in results],
                "scores": scores,
                "aggregation": self.aggregation,
            },
        )
