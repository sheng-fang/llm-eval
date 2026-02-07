"""Transcript recording for evaluation trials."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class TranscriptEntry:
    """
    A single entry in the transcript.

    Attributes:
        timestamp: When this entry was recorded
        entry_type: Type of entry (e.g., 'llm_call', 'tool_call', 'response')
        data: The actual data for this entry
    """

    timestamp: datetime
    entry_type: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "entry_type": self.entry_type,
            "data": self.data,
        }


class Transcript:
    """
    Records the complete interaction history of a trial.

    A transcript contains all LLM API calls, responses, tool calls,
    intermediate reasoning steps, and any other interactions during
    the evaluation trial.
    """

    def __init__(self) -> None:
        """Initialize an empty transcript."""
        self.entries: list[TranscriptEntry] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start(self) -> None:
        """Mark the start of recording."""
        self.start_time = datetime.now()

    def end(self) -> None:
        """Mark the end of recording."""
        self.end_time = datetime.now()

    def add_llm_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """
        Record an LLM API call.

        Args:
            model: Model identifier
            messages: Messages sent to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        self.entries.append(
            TranscriptEntry(
                timestamp=datetime.now(),
                entry_type="llm_call",
                data={
                    "model": model,
                    "messages": messages,
                    "parameters": kwargs,
                },
            )
        )

    def add_llm_response(
        self,
        response: str,
        usage: Optional[dict[str, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Record an LLM response.

        Args:
            response: The response text
            usage: Token usage information
            **kwargs: Additional response metadata
        """
        self.entries.append(
            TranscriptEntry(
                timestamp=datetime.now(),
                entry_type="llm_response",
                data={
                    "response": response,
                    "usage": usage or {},
                    "metadata": kwargs,
                },
            )
        )

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """
        Record a tool call.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
        """
        self.entries.append(
            TranscriptEntry(
                timestamp=datetime.now(),
                entry_type="tool_call",
                data={
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
            )
        )

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a tool result.

        Args:
            tool_name: Name of the tool
            result: Result returned by the tool
            error: Error message if the tool call failed
        """
        self.entries.append(
            TranscriptEntry(
                timestamp=datetime.now(),
                entry_type="tool_result",
                data={
                    "tool_name": tool_name,
                    "result": result,
                    "error": error,
                },
            )
        )

    def add_custom(
        self,
        entry_type: str,
        data: dict[str, Any],
    ) -> None:
        """
        Record a custom entry.

        Args:
            entry_type: Type identifier for the entry
            data: Entry data
        """
        self.entries.append(
            TranscriptEntry(
                timestamp=datetime.now(),
                entry_type=entry_type,
                data=data,
            )
        )

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the trial in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert transcript to dictionary."""
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert transcript to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        """
        Save transcript to a JSON file.

        Args:
            filepath: Path to save the transcript
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> "Transcript":
        """
        Load transcript from a JSON file.

        Args:
            filepath: Path to the transcript file

        Returns:
            Transcript instance
        """
        with open(filepath) as f:
            data = json.load(f)

        transcript = cls()
        if data["start_time"]:
            transcript.start_time = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            transcript.end_time = datetime.fromisoformat(data["end_time"])

        for entry_data in data["entries"]:
            transcript.entries.append(
                TranscriptEntry(
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    entry_type=entry_data["entry_type"],
                    data=entry_data["data"],
                )
            )

        return transcript
