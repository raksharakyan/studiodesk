"""ASGI middleware enforcing a maximum request body size.

Requests are rejected with 413 when the declared `Content-Length` exceeds the limit, and
streamed (chunked) bodies are cut off as soon as the running total passes it, so an
oversized body is never fully buffered.
"""

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TOO_LARGE_DETAIL = "Request body too large"
BAD_LENGTH_DETAIL = "Invalid Content-Length header"


class RequestBodyTooLarge(HTTPException):
    """Raised from the wrapped `receive` once the body exceeds the limit.

    It subclasses `HTTPException` so FastAPI re-raises it while parsing a body instead of
    turning it into a generic 400.
    """

    def __init__(self) -> None:
        """Create the 413 exception with a fixed, generic detail."""
        super().__init__(status_code=413, detail=TOO_LARGE_DETAIL)


class BodySizeLimitMiddleware:
    """Reject HTTP requests whose body is larger than `max_bytes` with 413."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        """Wrap `app`, allowing bodies of at most `max_bytes` bytes."""
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check the declared length, then enforce the limit on the streamed body."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared == -1:
            await _reject(scope, receive, send, 400, BAD_LENGTH_DETAIL)
            return
        if declared is not None and declared > self.max_bytes:
            await _reject(scope, receive, send, 413, TOO_LARGE_DETAIL)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge()
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await _reject(scope, receive, send, 413, TOO_LARGE_DETAIL)


def _declared_length(scope: Scope) -> int | None:
    """Return the Content-Length header as an int, None if absent, -1 if malformed."""
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            text = value.decode("latin-1").strip()
            if not text.isdigit():
                return -1
            return int(text)
    return None


async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
    """Send a JSON error response in FastAPI's `{"detail": ...}` shape."""
    response = JSONResponse({"detail": detail}, status_code=status)
    await response(scope, receive, send)
