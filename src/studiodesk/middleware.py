"""ASGI middleware: request body size limit, security headers, client-IP debug log.

`BodySizeLimitMiddleware`: requests are rejected with 413 when the declared
`Content-Length` exceeds the limit, and streamed (chunked) bodies are cut off as soon as
the running total passes it, so an oversized body is never fully buffered.
`SecurityHeadersMiddleware` adds baseline headers to every response;
`ClientIPDebugMiddleware` (opt-in) logs the peer and X-Forwarded-For only.
"""

import logging

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

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
            if not (text.isascii() and text.isdigit()):
                return -1
            return int(text)
    return None


async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
    """Send a JSON error response in FastAPI's `{"detail": ...}` shape."""
    response = JSONResponse({"detail": detail}, status_code=status)
    await response(scope, receive, send)


BASELINE_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
)
# Only in prod, where the app is served over HTTPS by the platform's edge proxy.
PROD_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
    (b"cross-origin-opener-policy", b"same-origin"),
)


class SecurityHeadersMiddleware:
    """Add baseline security headers to every HTTP response (and HSTS/COOP in prod).

    A header the route already set (e.g. the pages' own policy) is left unchanged, so
    page-specific values win over these defaults.
    """

    def __init__(self, app: ASGIApp, *, prod: bool) -> None:
        """Wrap `app`; with `prod`, also send HSTS and Cross-Origin-Opener-Policy."""
        self.app = app
        self.headers = BASELINE_HEADERS + (PROD_HEADERS if prod else ())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request through, adding missing headers to `http.response.start`."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers += [(n, v) for n, v in self.headers if n not in present]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


FORWARDED_FOR = b"x-forwarded-for"
MAX_LOGGED_FORWARDED_CHARS = 512


class ClientIPDebugMiddleware:
    """Log the direct peer and the raw X-Forwarded-For of each HTTP request at INFO.

    Only for working out a platform's proxy hops (to set TRUSTED_PROXY_IPS); enabled by
    DEBUG_LOG_CLIENT_IP. No other header, path or body is logged.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap `app`."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Log the peer host and X-Forwarded-For once, then pass the request through."""
        if scope["type"] == "http":
            client = scope.get("client")
            forwarded = ", ".join(
                value.decode("latin-1")
                for name, value in scope.get("headers", [])
                if name == FORWARDED_FOR
            )
            logger.info(
                "client ip debug",
                extra={
                    "peer": client[0] if client else None,
                    "x_forwarded_for": forwarded[:MAX_LOGGED_FORWARDED_CHARS] or None,
                },
            )
        await self.app(scope, receive, send)
