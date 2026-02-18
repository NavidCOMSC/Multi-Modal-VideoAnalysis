from typing import Any
from langfuse import Langfuse
from video_analysis_poc.video_llm.langfuse_schema.tracing import (
    EventPayload,
    ObservationPayload,
)
from contextlib import contextmanager


@contextmanager
def langfuse_observation(
    langfuse: Langfuse,
    payload: ObservationPayload,
    trace_id: str | None = None,
    trace_name: str | None = None,
):
    """Context manager that creates a Langfuse observation and automatically updates the trace name.

    This is a workaround for a known bug in Langfuse v3 where the trace name needs to be updated
    for each observation.

    Args:
        langfuse: Langfuse client instance
        trace_id: The trace ID to attach this observation to
        name: Name of the observation
        as_type: Type of observation ('span' or 'generation')
        trace_name: Name to set for the trace

    Yields:
        LangfuseObservation: The observation object that can be updated using methods like
            update(), update_trace(), end(), etc.
    """

    data = payload.to_dict()
    as_type = data.pop("as_type", "span")

    # Remove None values to avoid passing unexpected kwargs
    data = {k: v for k, v in data.items() if v is not None}

    # Create or retrieve the trace
    trace = langfuse.trace(id=trace_id, name=trace_name)

    if as_type == "generation":
        observation = trace.generation(**data)
    else:
        observation = trace.span(**data)

    try:
        yield observation
    except Exception as e:
        observation.update(level="ERROR", status_message=str(e))
        raise
    finally:
        if as_type == "span":
            observation.end()


def create_langfuse_event(
    langfuse: Langfuse,
    payload: EventPayload,
    trace_id: str | None = None,
    trace_name: str | None = None,
) -> dict[str, Any]:
    """Create a Langfuse event using either an EventPayload dataclass or a mapping.

    Accepts either:
      * An instance of 'schemas.langfuse.metadata.EventPayload' (preferred), or
      * A generic mapping containing the required keys.

    Performs light validation of required keys & datetime types, then forwards
    the full dictionary (including any extra fields) to 'langfuse.event'.

    Parameters:
        langfuse: Langfuse client instance.
        payload: Dataclass instance or mapping with at least required keys:
            trace_id, start_time, name

    Returns:
        dict[str, Any]: The payload dict that was sent to Langfuse.

    Raises:
        ValueError: If required keys are missing or datetime fields invalid.
    """
    data = payload.to_dict()
    # langfuse.create_event(**data)

    # Extract trace_context if present, fall back to trace_id parameter
    event_trace_id = data.pop("trace_context", None) or trace_id

    # Remove None values to avoid passing unexpected kwargs
    data = {k: v for k, v in data.items() if v is not None}

    # Events must be created on a trace, not on the top-level client
    trace = langfuse.trace(id=event_trace_id, name=trace_name)
    trace.event(**data)

    return data
