"""
Simple Q&A Evaluation Example

This example demonstrates basic usage of llm-eval for evaluating
a simple question-answering system.
"""

from llm_eval import ExactMatchGrader, Suite, Task
from llm_eval.graders.code_grader import ContainsGrader
from llm_eval.harness.base import SimpleHarness
from llm_eval.harness.executor import Executor
from llm_eval.metrics.aggregation import ResultAggregator


# Define a simple agent function
def simple_qa_agent(input_data):
    """
    A simple Q&A agent (mock implementation).

    In a real scenario, this would call an LLM API.
    """
    question = input_data.get("question", "")

    # Mock responses
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2+2?": "4",
        "Who wrote Romeo and Juliet?": "William Shakespeare",
    }

    return responses.get(question, "I don't know")


def main():
    # Create tasks
    tasks = [
        Task(
            id="capital_france",
            description="Answer question about France's capital",
            input_data={"question": "What is the capital of France?"},
            graders=[
                ExactMatchGrader(expected="Paris", case_sensitive=False),
            ],
            metadata={"category": "geography", "difficulty": "easy"},
        ),
        Task(
            id="math_addition",
            description="Simple addition problem",
            input_data={"question": "What is 2+2?"},
            graders=[
                ExactMatchGrader(expected="4"),
            ],
            metadata={"category": "math", "difficulty": "easy"},
        ),
        Task(
            id="shakespeare_author",
            description="Identify Shakespeare's work",
            input_data={"question": "Who wrote Romeo and Juliet?"},
            graders=[
                ContainsGrader(
                    expected_substrings=["Shakespeare"],
                    case_sensitive=False,
                ),
            ],
            metadata={"category": "literature", "difficulty": "easy"},
        ),
    ]

    # Create suite
    suite = Suite(
        name="Simple Q&A Evaluation",
        tasks=tasks,
        description="Basic question answering evaluation",
    )

    # Create harness and executor
    harness = SimpleHarness(agent_fn=simple_qa_agent)
    executor = Executor(harness=harness, max_workers=2, verbose=True)

    # Run evaluation with 3 trials per task
    results = executor.run_suite(suite, num_trials=3)

    # Analyze results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    aggregator = ResultAggregator(results)
    print(aggregator.summary_table())

    # Show failed trials
    failed = aggregator.failed_trials()
    if failed:
        print(f"\n{len(failed)} Failed Trials:")
        for trial in failed:
            print(f"  - {trial.task_id} (trial {trial.trial_number})")
            for gr in trial.grader_results:
                if not gr.passed:
                    print(f"    {gr.grader_name}: {gr.feedback}")

    # Calculate pass@k metrics
    from llm_eval.metrics.pass_at_k import PassAtKCalculator

    calc = PassAtKCalculator()
    for result in results.results:
        calc.add_result(result.task_id, result.passed)

    print("\n" + "=" * 80)
    print("PASS@K METRICS")
    print("=" * 80)
    print(f"pass@1: {calc.calculate_overall(k=1):.1%}")
    print(f"pass@2: {calc.calculate_overall(k=2):.1%}")
    print(f"pass@3: {calc.calculate_overall(k=3):.1%}")


if __name__ == "__main__":
    main()
