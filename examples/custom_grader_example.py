"""
Custom Grader Example

This example shows how to create custom graders for
domain-specific evaluation needs.
"""

from typing import Any, Optional

from llm_eval import Suite, Task
from llm_eval.graders.base import Grader, GraderResult
from llm_eval.harness.base import SimpleHarness
from llm_eval.harness.executor import Executor


class SentimentGrader(Grader):
    """
    Custom grader that checks sentiment of text.

    This is a simplified example - in practice, you'd use
    a sentiment analysis model.
    """

    def __init__(self, expected_sentiment: str, name: Optional[str] = None):
        """
        Initialize sentiment grader.

        Args:
            expected_sentiment: Expected sentiment ('positive', 'negative', 'neutral')
            name: Optional custom name
        """
        super().__init__(name)
        self.expected_sentiment = expected_sentiment.lower()

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade based on sentiment."""
        text = str(output).lower()

        # Simple keyword-based sentiment detection
        positive_words = ["good", "great", "excellent", "happy", "love", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "poor"]

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            detected_sentiment = "positive"
        elif neg_count > pos_count:
            detected_sentiment = "negative"
        else:
            detected_sentiment = "neutral"

        passed = detected_sentiment == self.expected_sentiment
        score = 1.0 if passed else 0.0

        feedback = f"Expected {self.expected_sentiment}, detected {detected_sentiment}"

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details={
                "expected": self.expected_sentiment,
                "detected": detected_sentiment,
                "positive_words": pos_count,
                "negative_words": neg_count,
            },
        )


class LengthGrader(Grader):
    """
    Custom grader that checks output length.
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize length grader.

        Args:
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            name: Optional custom name
        """
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length

    def grade(
        self,
        output: Any,
        expected: Any = None,
        transcript: Optional[Any] = None,
        **kwargs: Any,
    ) -> GraderResult:
        """Grade based on length."""
        text = str(output)
        length = len(text)

        passed = True
        feedback_parts = []

        if self.min_length is not None and length < self.min_length:
            passed = False
            feedback_parts.append(f"Too short (min: {self.min_length})")

        if self.max_length is not None and length > self.max_length:
            passed = False
            feedback_parts.append(f"Too long (max: {self.max_length})")

        if passed:
            feedback = f"Length OK ({length} chars)"
        else:
            feedback = ", ".join(feedback_parts) + f" (actual: {length})"

        # Partial credit based on how close to acceptable range
        if self.min_length and self.max_length:
            if length < self.min_length:
                score = max(0.0, length / self.min_length)
            elif length > self.max_length:
                score = max(0.0, 1.0 - (length - self.max_length) / self.max_length)
            else:
                score = 1.0
        else:
            score = 1.0 if passed else 0.0

        return GraderResult(
            grader_name=self.name,
            passed=passed,
            score=score,
            feedback=feedback,
            details={"length": length},
        )


# Mock agent
def review_agent(input_data):
    """Mock agent that generates product reviews."""
    product = input_data.get("product", "")
    sentiment = input_data.get("sentiment", "positive")

    if sentiment == "positive":
        return f"This {product} is excellent! I love it. Great quality and wonderful experience."
    else:
        return f"This {product} is terrible. I hate it. Poor quality and awful experience."


def main():
    # Create tasks with custom graders
    tasks = [
        Task(
            id="positive_review",
            description="Generate positive product review",
            input_data={"product": "laptop", "sentiment": "positive"},
            graders=[
                SentimentGrader(expected_sentiment="positive"),
                LengthGrader(min_length=20, max_length=200),
            ],
            metadata={"category": "content_generation"},
        ),
        Task(
            id="negative_review",
            description="Generate negative product review",
            input_data={"product": "phone", "sentiment": "negative"},
            graders=[
                SentimentGrader(expected_sentiment="negative"),
                LengthGrader(min_length=20, max_length=200),
            ],
            metadata={"category": "content_generation"},
        ),
    ]

    # Create suite
    suite = Suite(
        name="Custom Grader Demo",
        tasks=tasks,
        description="Demonstration of custom graders",
    )

    # Run evaluation
    harness = SimpleHarness(agent_fn=review_agent)
    executor = Executor(harness=harness, verbose=True)

    results = executor.run_suite(suite, num_trials=3)

    # Show results
    print("\n" + "=" * 80)
    print("CUSTOM GRADER RESULTS")
    print("=" * 80)
    print(results.summary())

    # Show detailed grader feedback
    print("\n" + "=" * 80)
    print("DETAILED GRADER FEEDBACK")
    print("=" * 80)

    from llm_eval.metrics.aggregation import ResultAggregator

    aggregator = ResultAggregator(results)

    for task_id, task_results in aggregator.by_task().items():
        print(f"\n{task_id}:")
        for trial in task_results[:1]:  # Show first trial
            print(f"  Output: {trial.output[:100]}...")
            for gr in trial.grader_results:
                print(f"  {gr.grader_name}:")
                print(f"    Score: {gr.score:.2f}")
                print(f"    Feedback: {gr.feedback}")
                if gr.details:
                    print(f"    Details: {gr.details}")


if __name__ == "__main__":
    main()
