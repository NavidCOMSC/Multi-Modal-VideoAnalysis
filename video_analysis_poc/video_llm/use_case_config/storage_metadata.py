from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model_name: str
    dimensions: int
    api_version: str


@dataclass
class PDFChunkingDataTemplate:
    separators: list[str]
    chunk_size: int
    chunk_overlap: int
    keep_separator: bool = True
    is_separator_regex: bool = True


@dataclass
class VectorDataStorageTemplate:
    vector_store_id: str
    embedding_model: EmbeddingModelConfig
    chunking: dict[Literal["pdf"], PDFChunkingDataTemplate]
    partition_key_config: dict
    distance_function: str = "cosine"
    data_type: str = "float32"


@dataclass
class VectorDataStorageConfig:
    database_name: str
    embedding_model_name: str
    api_version: str
    chunking: dict[Literal["pdf"], PDFChunkingDataTemplate]
    partition_key: dict
    vector_embedding_policy: dict
    indexing_policy: dict
    vector_search_fields: dict
    index_name: str
    text_key: str
    embedding_key: str
