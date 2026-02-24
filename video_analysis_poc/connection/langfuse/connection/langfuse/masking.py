from typing import Any


def pii_masker(data: Any, **kwargs) -> Any:
    if not isinstance(data, str):
        data = str(data)
    anonymised_text = "[REDACTED] *This is a placeholder for a robust anonymiser*"
    return anonymised_text
