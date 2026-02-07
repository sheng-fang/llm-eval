"""
Specialized Graders Example

Demonstrates the new specialized graders for various use cases.
"""

from llm_eval import (
    FormatGrader,
    LengthRangeGrader,
    NumericGrader,
    SimilarityGrader,
    Suite,
    Task,
)
from llm_eval.harness.base import SimpleHarness
from llm_eval.harness.executor import Executor


# Mock agents for different tasks
def math_agent(input_data):
    """Agent that solves math problems."""
    problem = input_data.get("problem", "")

    if "2+2" in problem:
        return "The answer is 4"
    elif "pi" in problem.lower():
        return "3.14159"
    elif "sqrt(16)" in problem:
        return "4.0"
    else:
        return "42"


def json_agent(input_data):
    """Agent that generates JSON responses."""
    request = input_data.get("request", "")

    if "user" in request.lower():
        return '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
    else:
        return '{"status": "success", "data": []}'


def paraphrase_agent(input_data):
    """Agent that paraphrases text."""
    text = input_data.get("text", "")

    # Simple paraphrasing (in reality, would use LLM)
    paraphrases = {
        "The quick brown fox jumps over the lazy dog": "A fast brown fox leaps over a sleepy dog",
        "Hello world": "Greetings, world",
    }

    return paraphrases.get(text, "Paraphrased version of the text")


def main():
    # Create tasks with specialized graders
    tasks = [
        # Numeric grading
        Task(
            id="math_pi",
            description="Calculate pi to 5 decimal places",
            input_data={"problem": "What is pi to 5 decimal places?"},
            graders=[
                NumericGrader(
                    expected=3.14159,
                    tolerance=0.00001,
                    name="PiGrader",
                )
            ],
            metadata={"category": "math"},
        ),
        # Similarity grading
        Task(
            id="paraphrase_fox",
            description="Paraphrase a sentence",
            input_data={"text": "The quick brown fox jumps over the lazy dog"},
            graders=[
                SimilarityGrader(
                    expected="A fast brown fox leaps over a sleepy dog",
                    threshold=0.5,
                    algorithm="sequence_matcher",
                    case_sensitive=False,
                )
            ],
            metadata={"category": "nlp"},
        ),
        # Format grading (JSON)
        Task(
            id="json_user",
            description="Generate user JSON",
            input_data={"request": "Generate a user object"},
            graders=[
                FormatGrader(format_type="json"),
                LengthRangeGrader(
                    min_length=20,
                    max_length=200,
                    unit="chars",
                ),
            ],
            metadata={"category": "structured_output"},
        ),
        # Multiple numeric tests
        Task(
            id="math_sqrt",
            description="Calculate square root",
            input_data={"problem": "What is sqrt(16)?"},
            graders=[
                NumericGrader(
                    expected=4.0,
                    tolerance=0.1,
                    relative=False,
                )
            ],
            metadata={"category": "math"},
        ),
    ]

    # Create suite
    suite = Suite(
        name="Specialized Graders Demo",
        tasks=tasks,
        description="Demonstration of specialized graders",
    )

    # Run evaluation
    print("\n" + "=" * 80)
    print("SPECIALIZED GRADERS DEMONSTRATION")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  • NumericGrader: Grading numeric outputs with tolerance")
    print("  • SimilarityGrader: Fuzzy string matching")
    print("  • FormatGrader: Validating output formats (JSON, XML, etc.)")
    print("  • LengthRangeGrader: Ensuring appropriate response length")
    print()

    # Use different agents for different tasks
    def multi_agent(input_data):
        """Route to appropriate agent based on task."""
        problem = input_data.get("problem", "")
        request = input_data.get("request", "")
        text = input_data.get("text", "")

        if problem:
            return math_agent(input_data)
        elif request:
            return json_agent(input_data)
        elif text:
            return paraphrase_agent(input_data)
        else:
            return "No response"

    harness = SimpleHarness(agent_fn=multi_agent)
    executor = Executor(harness=harness, verbose=True)

    results = executor.run_suite(suite, num_trials=3)

    # Show results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results.summary())

    from llm_eval.metrics.aggregation import ResultAggregator

    aggregator = ResultAggregator(results)
    print("\n" + aggregator.summary_table())

    # Show detailed grader feedback
    print("\n" + "=" * 80)
    print("DETAILED GRADER FEEDBACK")
    print("=" * 80)

    for task_id, task_results in aggregator.by_task().items():
        print(f"\n{task_id}:")
        trial = task_results[0]  # First trial
        print(f"  Output: {trial.output}")
        for gr in trial.grader_results:
            print(f"  {gr.grader_name}:")
            print(f"    Score: {gr.score:.2f}")
            print(f"    Feedback: {gr.feedback}")
            if gr.details:
                print(f"    Details: {gr.details}")


if __name__ == "__main__":
    main()
