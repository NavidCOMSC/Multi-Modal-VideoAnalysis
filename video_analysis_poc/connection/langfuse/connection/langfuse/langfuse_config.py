import os
from langfuse import Langfuse
from video_analysis_poc.video_llm.use_case_config.metadata import MetadataTemplate


def create_langfuse_client(metadata_instance: MetadataTemplate) -> Langfuse:
    """Create and return a configured Langfuse client.

    The client is configured using values from the provided MetadataTemplate,
    including host URL, public/secret keys, masking/anonymisation preference,
    and the current deployment environment (read from the ENV environment variable).

    Args:
        metadata_instance: Instance containing Langfuse connection details
            (metadata_instance.langfuse.host_url, public_key, secret_key, anonymise).

    Returns:
        Langfuse: An initialized Langfuse client ready for use.
    """
    if metadata_instance.langfuse is None:
        raise ValueError("Langfuse configuration is missing from metadata_instance.")

    langfuse_client = Langfuse(
            host=metadata_instance.langfuse.host_url,
            public_key=metadata_instance.langfuse.public_key,
            secret_key=metadata_instance.langfuse.secret_key,
            mask=metadata_instance.langfuse.anonymise,
            environment = os.getenv("ENV"))

    return langfuse_client