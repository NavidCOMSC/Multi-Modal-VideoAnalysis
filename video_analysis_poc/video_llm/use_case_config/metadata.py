from dataclasses import dataclass
from typing import Callable, Literal
from use_case_config.storage_metadata import VectorDataStorageConfig


class KeyvaultSecret(str):
    is_resolved: bool

    def __new__(cls, value: str, *, is_resolved: bool = False):
        obj = super().__new__(cls, value)
        obj.is_resolved = is_resolved
        return obj


@dataclass
class ArgusTemplate:
    client_id: KeyvaultSecret
    client_secret: KeyvaultSecret
    base_url: KeyvaultSecret
    argus_template_name: str
    output: Literal["llm_extraction", "ocr_extraction"] = "llm_extraction"


@dataclass
class RAGAgentTemplate:
    task_template_name: str  # e.g. highway_code, from this we have standard naming config, query gen prompt: highway_code_query_prompt, generation prompt: highway_code_generation_prompt
    vector_storage_config: VectorDataStorageConfig
    query_model_name: str
    generation_model_name: str
    api_version: str
    return_objects: int = 4  # Number of Document objects to return
    similarity_threshold: float = 0.5  # similarity score threshold for returns
    temperature: float | None = 1
    query_temperature: float | None = 1
    generation_temperature: float | None = 1
    query_reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "default"] | None
    ) = None
    generation_reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "default"] | None
    ) = None
    context_from_tasks: list[str] | None = None


@dataclass
class EmbeddingAgentTemplate:
    task_template_name: str
    model_name: str
    api_version: str


@dataclass
class LLMAgentTemplate:
    task_template_name: str  # Summerisation, q and a etc.
    model_name: str
    api_version: str
    temperature: float | None = 1
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "default"] | None
    ) = None
    context_from_tasks: list[str] | None = (
        None  # the names of other tasks required for this agents' context
    )


@dataclass
class LiteLLMTemplate:
    client_secret: KeyvaultSecret
    endpoint: KeyvaultSecret
    llm_result_task_name: str  # task name of the LLM required for the final output
    llm_agent: list[LLMAgentTemplate | RAGAgentTemplate]


@dataclass
class LangfuseTemplate:
    public_key: KeyvaultSecret
    secret_key: KeyvaultSecret
    host_url: str
    capture_input: bool = False  # prompt to define json output template
    capture_output: bool = False  # prompt to define json output template
    anonymise: Callable | None = None


@dataclass
class MetadataTemplate:
    argus: ArgusTemplate | None = None
    llm: LiteLLMTemplate | None = None
    langfuse: LangfuseTemplate | None = None
