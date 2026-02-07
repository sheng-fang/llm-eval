"""
Coding Agent Evaluation Example

This example demonstrates evaluating a coding agent with
unit tests and code quality checks.
"""

from llm_eval import Task, Suite, PythonAssertionGrader
from llm_eval.graders.code_grader import ContainsGrader
from llm_eval.harness.base import SimpleHarness
from llm_eval.harness.executor import Executor


# Mock coding agent
def coding_agent(input_data):
    """
    A mock coding agent that generates Python code.
    
    In reality, this would use an LLM to generate code.
    """
    task_description = input_data.get("task", "")
    
    # Mock code generation
    if "fibonacci" in task_description.lower():
        return """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    elif "factorial" in task_description.lower():
        return """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
    else:
        return "# Code not implemented"


def test_fibonacci_code(code):
    """Test if generated fibonacci code works."""
    try:
        # Execute the code
        namespace = {}
        exec(code, namespace)
        
        # Test the function
        if "fibonacci" not in namespace:
            return False
        
        fib = namespace["fibonacci"]
        
        # Test cases
        return (
            fib(0) == 0
            and fib(1) == 1
            and fib(5) == 5
            and fib(10) == 55
        )
    except Exception:
        return False


def test_factorial_code(code):
    """Test if generated factorial code works."""
    try:
        namespace = {}
        exec(code, namespace)
        
        if "factorial" not in namespace:
            return False
        
        fact = namespace["factorial"]
        
        return (
            fact(0) == 1
            and fact(1) == 1
            and fact(5) == 120
            and fact(10) == 3628800
        )
    except Exception:
        return False


def main():
    # Create coding tasks
    tasks = [
        Task(
            id="fibonacci_implementation",
            description="Implement Fibonacci function",
            input_data={
                "task": "Write a Python function to calculate the nth Fibonacci number"
            },
            graders=[
                ContainsGrader(
                    expected_substrings=["def fibonacci", "return"],
                    require_all=True,
                ),
                PythonAssertionGrader(
                    assertion_fn=test_fibonacci_code,
                    error_message="Fibonacci function tests failed",
                ),
            ],
            metadata={"category": "algorithms", "difficulty": "medium"},
        ),
        Task(
            id="factorial_implementation",
            description="Implement factorial function",
            input_data={
                "task": "Write a Python function to calculate factorial of n"
            },
            graders=[
                ContainsGrader(
                    expected_substrings=["def factorial", "return"],
                    require_all=True,
                ),
                PythonAssertionGrader(
                    assertion_fn=test_factorial_code,
                    error_message="Factorial function tests failed",
                ),
            ],
            metadata={"category": "algorithms", "difficulty": "easy"},
        ),
    ]

    # Create suite
    suite = Suite(
        name="Coding Agent Evaluation",
        tasks=tasks,
        description="Evaluate code generation capabilities",
    )

    # Run evaluation
    harness = SimpleHarness(agent_fn=coding_agent)
    executor = Executor(harness=harness, max_workers=2, verbose=True)
    
    results = executor.run_suite(suite, num_trials=5)

    # Show results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results.summary())

    # Detailed breakdown
    from llm_eval.metrics.aggregation import ResultAggregator
    
    aggregator = ResultAggregator(results)
    print("\n" + aggregator.summary_table())

    # Show which graders passed/failed
    print("\n" + "=" * 80)
    print("GRADER BREAKDOWN")
    print("=" * 80)
    
    for task_id, task_results in aggregator.by_task().items():
        print(f"\n{task_id}:")
        for trial in task_results:
            print(f"  Trial {trial.trial_number}: {'✓' if trial.passed else '✗'}")
            for gr in trial.grader_results:
                status = "✓" if gr.passed else "✗"
                print(f"    {status} {gr.grader_name}: {gr.score:.2f}")


if __name__ == "__main__":
    main()
