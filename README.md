# LLM Eval

A modular, extensible Python package for evaluating LLM applications and AI agents, based on best practices from Anthropic's evaluation framework.

## Features

- 🎯 **Flexible Task Management**: Define evaluation tasks with clear success criteria
- 🔍 **Multiple Grader Types**: Code-based, model-based (LLM-as-judge), and human graders
- 📊 **Comprehensive Metrics**: pass@k, pass^k, and custom metrics
- 🚀 **Parallel Execution**: Run evaluations concurrently with configurable parallelism
- 📝 **Rich Reporting**: Console, JSON, and HTML reports with transcript viewing
- 🔌 **Provider Agnostic**: Works with any LLM provider (OpenAI, Anthropic, local models)
- 🧩 **Extensible**: Easy to add custom graders, tasks, and reporters

## Installation

```bash
pip install llm-eval
```

For development with all optional dependencies:

```bash
pip install llm-eval[all]
```

Or install from source:

```bash
git clone https://github.com/yourusername/llm-eval.git
cd llm-eval
pip install -e .
```

## Quick Start

```python
from llm_eval import Task, Suite, ExactMatchGrader, RubricGrader
from llm_eval.harness import SimpleHarness

# Define a task
task = Task(
    id="greeting_test",
    description="Generate a friendly greeting",
    input_data={"prompt": "Say hello to the user"},
    graders=[
        ExactMatchGrader(expected="Hello!"),
        RubricGrader(rubric="Response should be friendly and concise")
    ]
)

# Create a suite
suite = Suite(name="basic_tests", tasks=[task])

# Run evaluation
harness = SimpleHarness(your_llm_function)
results = harness.run_suite(suite, num_trials=3)

# View results
print(results.summary())
```

## Core Concepts

Based on Anthropic's evaluation framework, this package uses the following concepts:

- **Task**: A single test with defined inputs and success criteria
- **Trial**: Each attempt at a task (multiple trials handle non-determinism)
- **Grader**: Logic that scores agent performance
- **Transcript**: Complete record of a trial (all interactions)
- **Suite**: Collection of related tasks
- **Harness**: Infrastructure that runs evaluations end-to-end

## Grader Types

### Code-based Graders (Deterministic)
- `ExactMatchGrader`: String exact matching
- `RegexGrader`: Pattern matching
- `JsonSchemaGrader`: JSON structure validation
- `PythonAssertionGrader`: Custom Python assertions
- `UnitTestGrader`: Run unit tests against output

### Model-based Graders (LLM-as-judge)
- `RubricGrader`: Score based on detailed rubrics
- `PairwiseGrader`: Compare two outputs
- `ReferenceGrader`: Compare against reference solution
- `MultiJudgeGrader`: Consensus from multiple LLM judges

### Human Graders
- `InteractiveGrader`: Interactive grading interface
- `BatchGrader`: Batch grading UI

## Metrics

- **pass@k**: Probability of at least 1 success in k attempts
- **pass^k**: Probability that all k trials succeed (consistency)
- Custom aggregation and statistical analysis

## Examples

See the `examples/` directory for complete examples:

- `simple_qa_eval.py`: Basic Q&A evaluation
- `coding_agent_eval.py`: Coding agent with unit tests
- `custom_grader_example.py`: Creating custom graders

## Documentation

For detailed documentation, see:
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Best Practices](docs/best_practices.md)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

This package is inspired by Anthropic's article ["Demystifying evals for AI agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) and incorporates best practices from the AI evaluation community.
