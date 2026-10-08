"""LLM access: the provider-neutral `LLMClient` protocol, typed errors and adapters.

Groq (JSON mode + Pydantic validation) is the default provider; Anthropic (structured
outputs with refusal fallback) is optional. `build_llm` picks one from Settings.
"""

from studiodesk.llm.anthropic_client import AnthropicLLM
from studiodesk.llm.base import (
    LLMClient,
    LLMError,
    LLMInvalidOutput,
    LLMRefusal,
    LLMTruncated,
    LLMUnavailable,
)
from studiodesk.llm.factory import build_llm
from studiodesk.llm.groq_client import GroqLLM

__all__ = [
    "AnthropicLLM",
    "GroqLLM",
    "LLMClient",
    "LLMError",
    "LLMInvalidOutput",
    "LLMRefusal",
    "LLMTruncated",
    "LLMUnavailable",
    "build_llm",
]
