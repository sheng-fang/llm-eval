"""pass@k and pass^k metrics implementation."""

import math
from typing import List


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    pass@k measures the probability that at least one of k samples passes.
    This uses an unbiased estimator.
    
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of samples to consider
        
    Returns:
        Estimated pass@k probability
        
    Reference:
        Chen et al. "Evaluating Large Language Models Trained on Code"
        https://arxiv.org/abs/2107.03374
    """
    if n - c < k:
        return 1.0

    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_power_k(success_rate: float, k: int) -> float:
    """
    Calculate pass^k metric.
    
    pass^k measures the probability that all k trials succeed.
    This is useful for measuring consistency.
    
    Args:
        success_rate: Per-trial success rate (0.0 to 1.0)
        k: Number of trials
        
    Returns:
        Probability that all k trials succeed
    """
    return success_rate**k


def calculate_pass_at_k_from_trials(
    trial_results: List[bool],
    k: int,
) -> float:
    """
    Calculate pass@k from a list of trial results.
    
    Args:
        trial_results: List of boolean trial results (True = passed)
        k: Number of samples to consider
        
    Returns:
        Estimated pass@k
    """
    n = len(trial_results)
    c = sum(trial_results)
    return pass_at_k(n, c, k)


def calculate_pass_power_k_from_trials(
    trial_results: List[bool],
    k: int,
) -> float:
    """
    Calculate pass^k from a list of trial results.
    
    Args:
        trial_results: List of boolean trial results (True = passed)
        k: Number of consecutive trials to consider
        
    Returns:
        Estimated pass^k
    """
    if len(trial_results) < k:
        return 0.0

    # Calculate empirical success rate
    success_rate = sum(trial_results) / len(trial_results)
    return pass_power_k(success_rate, k)


def confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for pass rate.
    
    Args:
        successes: Number of successful trials
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)

    p = successes / total
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


class PassAtKCalculator:
    """
    Helper class for calculating pass@k metrics across multiple tasks.
    """

    def __init__(self) -> None:
        """Initialize calculator."""
        self.task_results: dict[str, List[bool]] = {}

    def add_result(self, task_id: str, passed: bool) -> None:
        """
        Add a trial result.
        
        Args:
            task_id: Task identifier
            passed: Whether the trial passed
        """
        if task_id not in self.task_results:
            self.task_results[task_id] = []
        self.task_results[task_id].append(passed)

    def calculate(self, k: int = 1) -> dict[str, float]:
        """
        Calculate pass@k for all tasks.
        
        Args:
            k: Number of samples to consider
            
        Returns:
            Dictionary mapping task_id to pass@k score
        """
        results = {}
        for task_id, trial_results in self.task_results.items():
            results[task_id] = calculate_pass_at_k_from_trials(trial_results, k)
        return results

    def calculate_overall(self, k: int = 1) -> float:
        """
        Calculate overall pass@k across all tasks.
        
        Args:
            k: Number of samples to consider
            
        Returns:
            Average pass@k across all tasks
        """
        task_scores = self.calculate(k)
        if not task_scores:
            return 0.0
        return sum(task_scores.values()) / len(task_scores)

    def calculate_pass_power_k(self, k: int = 1) -> dict[str, float]:
        """
        Calculate pass^k for all tasks.
        
        Args:
            k: Number of consecutive trials
            
        Returns:
            Dictionary mapping task_id to pass^k score
        """
        results = {}
        for task_id, trial_results in self.task_results.items():
            results[task_id] = calculate_pass_power_k_from_trials(trial_results, k)
        return results
