from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass
class ObservationPayload:
    """
    Container for a single observation (span/generation/embedding) suitable for logging, analytics,
    or forwarding to an observability platform (e.g. Langfuse).

    This payload is used with the start_as_current_observation context manager to create
    and manage observations within a trace context.

    Attributes:
        name: Name of the observation (e.g., function or operation name).
        as_type: Type of observation to create ("span", "generation", or "embedding"). Defaults to "span".
        input: Input data for the operation (can be any JSON-serializable object).
        output: Output data from the operation (can be any JSON-serializable object).
        metadata: Additional metadata to associate with the observation.
        version: Version identifier for the code or component.
        level: Importance level of the observation (info, warning, error).
        status_message: Optional status message for the observation.
        completion_start_time: When the model started generating the response (for generation/embedding types).
        model: Name/identifier of the AI model used (e.g., "gpt-4") (for generation/embedding types).
        model_parameters: Parameters used for the model (e.g., temperature, max_tokens) (for generation/embedding types).
        usage_details: Token usage information (e.g., prompt_tokens, completion_tokens) (for generation/embedding types).
        cost_details: Cost information for the model call (for generation/embedding types).
        prompt: Associated prompt template from Langfuse prompt management (for generation/embedding types).
        end_on_exit: Whether to end the span automatically when leaving the context manager. Defaults to True.
    """

    name: str
    as_type: str = "span"
    input: Any | None = None
    output: Any | None = None
    metadata: Any | None = None
    version: str | None = None
    level: str | None = None
    status_message: str | None = None
    completion_start_time: datetime | None = None
    model: str | None = None
    model_parameters: dict[str, Any] | None = None
    usage_details: dict[str, int] | None = None
    cost_details: dict[str, float] | None = None
    prompt: Any | None = None
    end_on_exit: bool | None = True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ObservationPayload instance into a plain dictionary.

        Returns:
            A dict[str, Any] containing all dataclass fields (already JSON-serialisable
            unless a field value itself is not serialisable).
        """
        payload_dictionary = asdict(self)
        return payload_dictionary


@dataclass
class EventPayload:
    """
    Container for a single event (trace / span) suitable for logging, analytics,
    or forwarding to an observability platform (e.g. Langfuse).

    Attributes:
        trace_id: Correlation / trace identifier for grouping related events.
        start_time: UTC timestamp when the event started.
        name: Human readable name or label for this event (e.g. "rerank_pass", "summary").
        level : The level of the event. Can be DEBUG, DEFAULT, WARNING or ERROR. Used for sorting/filtering of traces with elevated error levels and for highlighting in the UI.
        status_message: The status message of the event. Additional field for context of the event. E.g. the error message of an error event.
        input: The input to the event. Can be any JSON object.
        output: The output to the event. Can be any JSON object.

    """

    name: str
    level: str | None = None
    status_message: str | None = None
    input: str | None = None
    output: str | None = None
    trace_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the EventPayload instance into a plain dictionary.

        Returns:
            A dict[str, Any] containing all dataclass fields (already JSON-serialisable
            unless a field value itself is not serialisable).
        """
        payload_dictionary = asdict(self)
        return payload_dictionary
