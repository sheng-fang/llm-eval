"""Human grading interfaces."""

from typing import Any, Optional

from llm_eval.graders.base import Grader, GraderResult


class HumanGrader(Grader):
    """
    Interactive human grading interface.

    Prompts a human to grade the output during evaluation.
    Useful for calibration and spot-checking.
    """

    def __init__(
        self,
        criteria: str = "overall quality",
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize human grader.

        Args:
            criteria: Grading criteria to show the human
            name: Optional custom name
        """
        super().__init__(name)
        self.criteria = criteria

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade output with human input."""
        print("\n" + "=" * 80)
        print("HUMAN GRADING REQUIRED")
        print("=" * 80)
        print(f"\nCriteria: {self.criteria}")
        print(f"\nOutput to grade:\n{output}")

        if expected is not None:
            print(f"\nExpected output:\n{expected}")

        print("\n" + "-" * 80)

        # Get score
        while True:
            try:
                score_input = input("Enter score (0.0 to 1.0): ").strip()
                score = float(score_input)
                if 0.0 <= score <= 1.0:
                    break
                print("Score must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Get feedback
        feedback = input("Enter feedback (optional): ").strip()

        passed = score >= 0.5

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback or f"Human score: {score}",
            details={"grading_method": "interactive"},
        )


class BatchHumanGrader:
    """
    Batch human grading interface.

    Collects outputs for batch grading rather than grading one at a time.
    More efficient for grading many outputs.
    """

    def __init__(self, criteria: str = "overall quality") -> None:
        """
        Initialize batch human grader.

        Args:
            criteria: Grading criteria
        """
        self.criteria = criteria
        self.pending_grades: list[dict[str, Any]] = []

    def add_for_grading(
        self,
        task_id: str,
        output: Any,
        expected: Any = None,
    ) -> None:
        """
        Add an output for batch grading.

        Args:
            task_id: Task identifier
            output: Output to grade
            expected: Expected output
        """
        self.pending_grades.append(
            {
                "task_id": task_id,
                "output": output,
                "expected": expected,
            }
        )

    def grade_batch(self) -> dict[str, GraderResult]:
        """
        Grade all pending outputs interactively.

        Returns:
            Dictionary mapping task_id to GraderResult
        """
        results = {}

        print("\n" + "=" * 80)
        print(f"BATCH HUMAN GRADING: {len(self.pending_grades)} outputs")
        print("=" * 80)

        for i, item in enumerate(self.pending_grades, 1):
            print(f"\n[{i}/{len(self.pending_grades)}] Task: {item['task_id']}")
            print(f"Criteria: {self.criteria}")
            print(f"\nOutput:\n{item['output']}")

            if item["expected"] is not None:
                print(f"\nExpected:\n{item['expected']}")

            print("\n" + "-" * 80)

            # Get score
            while True:
                try:
                    score_input = input("Score (0.0-1.0) or 's' to skip: ").strip()
                    if score_input.lower() == "s":
                        score = 0.5
                        feedback = "Skipped"
                        break
                    score = float(score_input)
                    if 0.0 <= score <= 1.0:
                        feedback = input("Feedback (optional): ").strip()
                        break
                    print("Score must be between 0.0 and 1.0")
                except ValueError:
                    print("Invalid input.")

            results[item["task_id"]] = GraderResult(
                grader_name="BatchHumanGrader",
                passed=score >= 0.5,
                score=score,
                feedback=feedback or f"Human score: {score}",
                details={"grading_method": "batch"},
            )

        self.pending_grades.clear()
        return results

    def export_for_crowdsourcing(self, filepath: str) -> None:
        """
        Export pending grades to a file for crowdsourcing.

        Args:
            filepath: Path to save the export
        """
        import json

        with open(filepath, "w") as f:
            json.dump(
                {
                    "criteria": self.criteria,
                    "items": self.pending_grades,
                },
                f,
                indent=2,
            )

        print(f"Exported {len(self.pending_grades)} items to {filepath}")
