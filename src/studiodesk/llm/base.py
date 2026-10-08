"""The provider-neutral `LLMClient` protocol and the typed errors every adapter raises."""

from typing import Protocol

from pydantic import BaseModel


class LLMError(RuntimeError):
    """Base class for LLM failures. Messages are safe to log, never to return to clients."""


class LLMUnavailable(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The provider could not be reached or rejected the call (rate limit, 5xx, timeout)."""


class LLMRefusal(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The model declined to answer."""


class LLMTruncated(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The response hit the token limit before the structured output was complete."""


class LLMInvalidOutput(LLMError):  # noqa: N818 (names fixed by the M3 spec)
    """The response did not contain JSON matching the requested schema."""


class LLMClient(Protocol):
    """Anything that can turn a system prompt plus user content into a schema instance."""

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Return the model's answer parsed as `schema`.

        Raises:
            LLMError: any failure (a subclass says which kind).
        """
        ...

    def close(self) -> None:
        """Release the underlying client's connections."""
        ...


def error_name(exc: BaseException) -> str:
    """Class name plus HTTP status (if any); never the message, which may echo the request."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}({status})" if isinstance(status, int) else type(exc).__name__
