"""Base harness interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from llm_eval.core.task import Task
from llm_eval.core.trial import Trial


class BaseHarness(ABC):
    """
    Abstract base class for evaluation harnesses.
    
    A harness is responsible for:
    1. Setting up the environment for each trial
    2. Running the agent on a task
    3. Capturing outputs and outcomes
    4. Cleaning up after each trial
    """

    @abstractmethod
    def setup(self, task: Task) -> None:
        """
        Set up the environment for a task.
        
        Args:
            task: Task to set up for
        """
        pass

    @abstractmethod
    def run_trial(self, task: Task, trial: Trial) -> None:
        """
        Run a single trial of a task.
        
        This method should:
        1. Call trial.start()
        2. Execute the agent
        3. Record interactions in trial.transcript
        4. Call trial.complete() or trial.fail()
        
        Args:
            task: Task to run
            trial: Trial instance to populate
        """
        pass

    @abstractmethod
    def teardown(self, task: Task) -> None:
        """
        Clean up after a task.
        
        Args:
            task: Task to clean up for
        """
        pass


class SimpleHarness(BaseHarness):
    """
    Simple harness for basic LLM evaluation.
    
    Runs a function that takes input and returns output.
    """

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, Any]], Any],
        capture_transcript: bool = True,
    ) -> None:
        """
        Initialize simple harness.
        
        Args:
            agent_fn: Function that takes input_data and returns output
            capture_transcript: Whether to capture transcript (requires agent_fn support)
        """
        self.agent_fn = agent_fn
        self.capture_transcript = capture_transcript

    def setup(self, task: Task) -> None:
        """No setup needed for simple harness."""
        pass

    def run_trial(self, task: Task, trial: Trial) -> None:
        """Run a trial using the agent function."""
        trial.start()

        try:
            # Run the agent
            output = self.agent_fn(task.input_data)

            # Complete the trial
            trial.complete(output=output)

        except Exception as e:
            trial.fail(str(e))

    def teardown(self, task: Task) -> None:
        """No teardown needed for simple harness."""
        pass


class CallbackHarness(BaseHarness):
    """
    Harness with custom setup/teardown callbacks.
    
    Useful for more complex scenarios requiring environment preparation.
    """

    def __init__(
        self,
        agent_fn: Callable[[Dict[str, Any]], Any],
        setup_fn: Optional[Callable[[Task], None]] = None,
        teardown_fn: Optional[Callable[[Task], None]] = None,
    ) -> None:
        """
        Initialize callback harness.
        
        Args:
            agent_fn: Function that takes input_data and returns output
            setup_fn: Optional setup function
            teardown_fn: Optional teardown function
        """
        self.agent_fn = agent_fn
        self.setup_fn = setup_fn
        self.teardown_fn = teardown_fn

    def setup(self, task: Task) -> None:
        """Run setup callback if provided."""
        if self.setup_fn:
            self.setup_fn(task)

    def run_trial(self, task: Task, trial: Trial) -> None:
        """Run a trial using the agent function."""
        trial.start()

        try:
            output = self.agent_fn(task.input_data)
            trial.complete(output=output)
        except Exception as e:
            trial.fail(str(e))

    def teardown(self, task: Task) -> None:
        """Run teardown callback if provided."""
        if self.teardown_fn:
            self.teardown_fn(task)
