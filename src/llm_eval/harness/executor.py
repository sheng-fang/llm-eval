"""Parallel execution engine for running evaluations."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, TaskID

from llm_eval.core.suite import Suite
from llm_eval.core.task import Task
from llm_eval.core.trial import Trial, TrialResult
from llm_eval.harness.base import BaseHarness


@dataclass
class EvaluationResult:
    """
    Results from running an evaluation suite.
    
    Attributes:
        suite_name: Name of the suite
        results: List of all trial results
        start_time: When evaluation started
        end_time: When evaluation ended
        metadata: Additional metadata
    """

    suite_name: str
    results: List[TrialResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get evaluation duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def num_tasks(self) -> int:
        """Get number of unique tasks."""
        return len(set(r.task_id for r in self.results))

    @property
    def num_trials(self) -> int:
        """Get total number of trials."""
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        """Get overall pass rate."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passed)
        return passed / len(self.results)

    @property
    def average_score(self) -> float:
        """Get average score across all trials."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def get_task_results(self, task_id: str) -> List[TrialResult]:
        """Get all results for a specific task."""
        return [r for r in self.results if r.task_id == task_id]

    def summary(self) -> str:
        """Get a summary string of the results."""
        lines = [
            f"Evaluation Results: {self.suite_name}",
            f"=" * 60,
            f"Tasks: {self.num_tasks}",
            f"Trials: {self.num_trials}",
            f"Pass Rate: {self.pass_rate:.1%}",
            f"Average Score: {self.average_score:.3f}",
            f"Duration: {self.duration:.2f}s" if self.duration else "Duration: N/A",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "suite_name": self.suite_name,
            "num_tasks": self.num_tasks,
            "num_trials": self.num_trials,
            "pass_rate": self.pass_rate,
            "average_score": self.average_score,
            "duration": self.duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }


class Executor:
    """
    Parallel execution engine for running evaluations.
    
    Handles concurrent task execution, multiple trials per task,
    progress tracking, and result aggregation.
    """

    def __init__(
        self,
        harness: BaseHarness,
        max_workers: int = 4,
        verbose: bool = True,
    ) -> None:
        """
        Initialize executor.
        
        Args:
            harness: Harness to use for running trials
            max_workers: Maximum number of parallel workers
            verbose: Whether to show progress
        """
        self.harness = harness
        self.max_workers = max_workers
        self.verbose = verbose
        self.console = Console() if verbose else None

    def run_suite(
        self,
        suite: Suite,
        num_trials: int = 1,
    ) -> EvaluationResult:
        """
        Run an entire evaluation suite.
        
        Args:
            suite: Suite to run
            num_trials: Number of trials per task
            
        Returns:
            EvaluationResult with all trial results
        """
        eval_result = EvaluationResult(suite_name=suite.name)
        eval_result.start_time = datetime.now()

        if self.verbose and self.console:
            self.console.print(f"\n[bold]Running suite: {suite.name}[/bold]")
            self.console.print(f"Tasks: {suite.num_tasks}, Trials per task: {num_trials}\n")

        # Run all tasks
        for task in suite.tasks:
            task_results = self.run_task(task, num_trials)
            eval_result.results.extend(task_results)

        eval_result.end_time = datetime.now()

        if self.verbose and self.console:
            self.console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")
            self.console.print(eval_result.summary())

        return eval_result

    def run_task(
        self,
        task: Task,
        num_trials: int = 1,
    ) -> List[TrialResult]:
        """
        Run a single task with multiple trials.
        
        Args:
            task: Task to run
            num_trials: Number of trials
            
        Returns:
            List of TrialResults
        """
        # Setup
        self.harness.setup(task)

        results: List[TrialResult] = []

        try:
            if self.verbose and self.console:
                with Progress() as progress:
                    task_progress = progress.add_task(
                        f"[cyan]{task.id}",
                        total=num_trials,
                    )

                    # Run trials in parallel
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        for trial_num in range(num_trials):
                            future = executor.submit(self._run_single_trial, task, trial_num)
                            futures.append(future)

                        for future in as_completed(futures):
                            trial_result = future.result()
                            results.append(trial_result)
                            progress.update(task_progress, advance=1)
            else:
                # Run without progress bar
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(self._run_single_trial, task, i)
                        for i in range(num_trials)
                    ]
                    results = [f.result() for f in as_completed(futures)]

        finally:
            # Teardown
            self.harness.teardown(task)

        return results

    def _run_single_trial(self, task: Task, trial_number: int) -> TrialResult:
        """Run a single trial and grade it."""
        trial = Trial(task.id, trial_number)

        # Run the trial
        self.harness.run_trial(task, trial)

        # Grade the trial
        grader_results = []
        for grader in task.graders:
            grader_result = grader.grade(
                output=trial.output,
                expected=task.reference_solution,
                transcript=trial.transcript,
            )
            grader_results.append(grader_result)

        # Convert to result
        return trial.to_result(grader_results)
