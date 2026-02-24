from langfuse import Langfuse
from langfuse.model import TextPromptClient
import os
from typing import Literal, Any
from connection.langfuse.constants import enviornment_dict
from video_analysis_poc.video_llm.use_case_config import MetadataTemplate
import json


async def collate_prompt(
    metadata_instance: MetadataTemplate,
    langfuse: Langfuse,
    prompt_type: Literal["task", "argus"],
    prompt_template_name: str,
    uuid_tag: str | None = None,
    user_query: str | None = None,
    argus_llm_context: list[tuple[str, str]] | None = None,
    argus_ocr_context: list[tuple[str, str]] | None = None,
    **prompt_kwargs,
) -> tuple[str | dict[str, Any], TextPromptClient]:
    """
    Compile a Langfuse prompt.

    Args:
        metadata_instance: Metadata with template names and Argus client id.
        langfuse: Langfuse client.
        prompt_type: "task" or "argus".
        prompt_template_name: Name of the prompt template to use.
        uuid_tag: Optional correlation / tracing identifier.
        user_query: Optional user input.
        argus_llm_context: Optional list of (key, value) pairs injected into the template from ARGUS LLM.
        argus_ocr_context: Optional list of (key, value) pairs injected into the template from ARGUS OCR.
        **prompt_kwargs: Additional keyword arguments to pass to the prompt template compilation.

    Returns:
        (compiled_prompt, prompt_object)
        - compiled_prompt: str for task, dict for argus.
        - prompt_object (TextPromptClient):
            Represents the resolved Langfuse prompt entity (name + label + version).
            Useful for:
              * Inspecting template metadata (e.g. name, label, version, id)
              * Auditing which prompt version was used in a run
              * Re-compiling with different dynamic values if needed

    Raises:
        KeyError: Missing or unmapped ENV.
        ValueError: Invalid prompt_type or malformed context entry.
        json.JSONDecodeError: Argus prompt not valid JSON.
    """

    # TODO: check that defaulting on dev is acceptable prompt_env = enviornment_dict[os.getenv("ENV", "dev")]

    env = os.getenv("ENV")
    if env is None:
        raise ValueError("Environment variable 'ENV' is not set.")
    prompt_env = enviornment_dict[env]

    prompt_object = langfuse.get_prompt(name=prompt_template_name, label=prompt_env)

    if metadata_instance.argus is None:
        raise ValueError("metadata_instance.argus is not set.")

    compiled_prompt = prompt_object.compile(
        argus_client_id=metadata_instance.argus.client_id,
        prompt_template_name=prompt_template_name,
        argus_uuid_tag=uuid_tag,
        argus_llm_context=argus_llm_context,
        argus_ocr_context=argus_ocr_context,
        user_query=user_query,
        **prompt_kwargs,
    )

    if prompt_type == "argus":
        compiled_prompt = json.loads(compiled_prompt)

    return compiled_prompt, prompt_object
