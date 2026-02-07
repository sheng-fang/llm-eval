"""Result aggregation and analysis utilities."""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from llm_eval.core.trial import TrialResult
from llm_eval.harness.executor import EvaluationResult


class ResultAggregator:
    """
    Aggregates and analyzes evaluation results.
    
    Provides statistics, breakdowns by category, and comparison utilities.
    """

    def __init__(self, results: EvaluationResult) -> None:
        """
        Initialize aggregator.
        
        Args:
            results: Evaluation results to analyze
        """
        self.results = results

    def by_task(self) -> Dict[str, List[TrialResult]]:
        """Group results by task."""
        grouped: Dict[str, List[TrialResult]] = defaultdict(list)
        for result in self.results.results:
            grouped[result.task_id].append(result)
        return dict(grouped)

    def task_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each task.
        
        Returns:
            Dictionary mapping task_id to statistics
        """
        stats = {}
        for task_id, task_results in self.by_task().items():
            passed = sum(1 for r in task_results if r.passed)
            scores = [r.score for r in task_results]

            stats[task_id] = {
                "num_trials": len(task_results),
                "num_passed": passed,
                "pass_rate": passed / len(task_results) if task_results else 0.0,
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
            }

        return stats

    def summary_table(self) -> str:
        """
        Generate a summary table of results.
        
        Returns:
            Formatted table string
        """
        stats = self.task_statistics()

        lines = [
            "Task Summary",
            "=" * 80,
            f"{'Task ID':<30} {'Trials':<10} {'Passed':<10} {'Pass Rate':<12} {'Avg Score':<10}",
            "-" * 80,
        ]

        for task_id, task_stats in stats.items():
            lines.append(
                f"{task_id:<30} "
                f"{task_stats['num_trials']:<10} "
                f"{task_stats['num_passed']:<10} "
                f"{task_stats['pass_rate']:<12.1%} "
                f"{task_stats['average_score']:<10.3f}"
            )

        lines.append("-" * 80)
        lines.append(
            f"{'OVERALL':<30} "
            f"{self.results.num_trials:<10} "
            f"{sum(1 for r in self.results.results if r.passed):<10} "
            f"{self.results.pass_rate:<12.1%} "
            f"{self.results.average_score:<10.3f}"
        )

        return "\n".join(lines)

    def failed_trials(self) -> List[TrialResult]:
        """Get all failed trials."""
        return [r for r in self.results.results if not r.passed]

    def passed_trials(self) -> List[TrialResult]:
        """Get all passed trials."""
        return [r for r in self.results.results if r.passed]


def compare_results(
    baseline: EvaluationResult,
    current: EvaluationResult,
) -> Dict[str, Any]:
    """
    Compare two evaluation results (e.g., before/after a change).
    
    Args:
        baseline: Baseline evaluation results
        current: Current evaluation results
        
    Returns:
        Comparison statistics
    """
    baseline_agg = ResultAggregator(baseline)
    current_agg = ResultAggregator(current)

    baseline_stats = baseline_agg.task_statistics()
    current_stats = current_agg.task_statistics()

    # Find common tasks
    common_tasks = set(baseline_stats.keys()) & set(current_stats.keys())

    improvements = []
    regressions = []
    unchanged = []

    for task_id in common_tasks:
        baseline_rate = baseline_stats[task_id]["pass_rate"]
        current_rate = current_stats[task_id]["pass_rate"]
        delta = current_rate - baseline_rate

        if abs(delta) < 0.01:  # Less than 1% change
            unchanged.append(task_id)
        elif delta > 0:
            improvements.append((task_id, delta))
        else:
            regressions.append((task_id, delta))

    return {
        "baseline_pass_rate": baseline.pass_rate,
        "current_pass_rate": current.pass_rate,
        "delta": current.pass_rate - baseline.pass_rate,
        "num_improvements": len(improvements),
        "num_regressions": len(regressions),
        "num_unchanged": len(unchanged),
        "improvements": sorted(improvements, key=lambda x: x[1], reverse=True),
        "regressions": sorted(regressions, key=lambda x: x[1]),
    }


def detect_regressions(
    baseline: EvaluationResult,
    current: EvaluationResult,
    threshold: float = 0.05,
) -> List[str]:
    """
    Detect tasks that regressed significantly.
    
    Args:
        baseline: Baseline evaluation results
        current: Current evaluation results
        threshold: Minimum pass rate decrease to consider a regression
        
    Returns:
        List of task IDs that regressed
    """
    comparison = compare_results(baseline, current)
    return [task_id for task_id, delta in comparison["regressions"] if abs(delta) >= threshold]
