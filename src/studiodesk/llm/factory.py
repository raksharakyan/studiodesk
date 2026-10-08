"""Pick and build the configured LLM provider."""

import logging

from studiodesk.config import Settings
from studiodesk.llm.anthropic_client import AnthropicLLM
from studiodesk.llm.base import LLMClient
from studiodesk.llm.groq_client import GroqLLM

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLMClient | None:
    """Build the client for `settings.llm_provider`, or None if its API key is missing.

    No network call is made. A missing key is logged once as a warning; the API then
    answers LLM routes with 503 "LLM not configured".
    """
    if settings.llm_api_key is None:
        logger.warning(
            "llm not configured: no API key for the selected provider",
            extra={"provider": settings.llm_provider},
        )
        return None
    if settings.llm_provider == "anthropic":
        return AnthropicLLM.from_settings(settings)
    return GroqLLM.from_settings(settings)
