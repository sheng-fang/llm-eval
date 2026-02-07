"""
Async LLM Grading Example

Demonstrates async support for LLM-based graders with parallel execution.
"""

import asyncio
from llm_eval import Task, Suite
from llm_eval.graders.model_grader import RubricGrader, MultiJudgeGrader
from llm_eval.harness.base import SimpleHarness
from llm_eval.harness.executor import Executor


# Mock async LLM function
async def mock_async_llm(prompt: str) -> str:
    """
    Mock async LLM function that simulates API call delay.
    
    In production, this would be an actual async LLM API call
    (e.g., using aiohttp with OpenAI or Anthropic).
    """
    # Simulate API latency
    await asyncio.sleep(0.1)
    
    # Mock response based on prompt content
    if "concise" in prompt.lower():
        return """SCORE: 0.9
FEEDBACK: The response is clear and concise, effectively communicating the main point without unnecessary verbosity."""
    elif "detailed" in prompt.lower():
        return """SCORE: 0.85
FEEDBACK: The response provides good detail and covers the topic comprehensively. Could benefit from more specific examples."""
    else:
        return """SCORE: 0.75
FEEDBACK: The response is acceptable but could be improved with better structure and clarity."""


# Mock agent
def summary_agent(input_data):
    """Mock agent that generates summaries."""
    topic = input_data.get("topic", "")
    style = input_data.get("style", "concise")
    
    if style == "concise":
        return f"Brief overview of {topic}: Key points covered efficiently."
    else:
        return f"Detailed analysis of {topic}: This comprehensive review examines multiple aspects, providing in-depth coverage of the subject matter with extensive examples and thorough explanations."


async def main():
    """Main async function to run the evaluation."""
    
    # Create tasks with async LLM graders
    tasks = [
        Task(
            id="concise_summary",
            description="Generate concise summary",
            input_data={"topic": "climate change", "style": "concise"},
            graders=[
                RubricGrader(
                    rubric="Response should be concise and clear",
                    llm_fn=mock_async_llm,
                    is_async=True,
                    model="mock-async-llm",
                )
            ],
            metadata={"category": "summarization"},
        ),
        Task(
            id="detailed_summary",
            description="Generate detailed summary",
            input_data={"topic": "artificial intelligence", "style": "detailed"},
            graders=[
                RubricGrader(
                    rubric="Response should be detailed and comprehensive",
                    llm_fn=mock_async_llm,
                    is_async=True,
                    model="mock-async-llm",
                )
            ],
            metadata={"category": "summarization"},
        ),
    ]

    # Create suite
    suite = Suite(
        name="Async LLM Grading Demo",
        tasks=tasks,
        description="Demonstration of async LLM-based grading",
    )

    # Run evaluation
    harness = SimpleHarness(agent_fn=summary_agent)
    executor = Executor(harness=harness, verbose=True)

    print("\n" + "=" * 80)
    print("ASYNC LLM GRADING DEMONSTRATION")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  • Async LLM grader execution")
    print("  • Parallel grading for better performance")
    print("  • Integration with evaluation harness")
    print()

    results = executor.run_suite(suite, num_trials=3)

    # Show results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results.summary())

    from llm_eval.metrics.aggregation import ResultAggregator
    
    aggregator = ResultAggregator(results)
    print("\n" + aggregator.summary_table())

    # Show grader feedback
    print("\n" + "=" * 80)
    print("LLM GRADER FEEDBACK")
    print("=" * 80)
    
    for task_id, task_results in aggregator.by_task().items():
        print(f"\n{task_id}:")
        for trial in task_results[:1]:  # Show first trial
            for gr in trial.grader_results:
                print(f"  Model: {gr.details.get('model', 'N/A')}")
                print(f"  Score: {gr.score:.2f}")
                print(f"  Feedback: {gr.feedback[:200]}...")


def run_sync_example():
    """
    Synchronous wrapper for running the async example.
    
    This allows the example to be run from a non-async context.
    """
    asyncio.run(main())


if __name__ == "__main__":
    run_sync_example()
