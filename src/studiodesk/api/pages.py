"""HTML pages and static assets, served with strict security headers.

`GET /voice` serves `static/voice.html`; `/static` serves the rest of `static/`. Both carry
a Content-Security-Policy that allows scripts only from this origin and the one pinned SDK
file on the CDN (`ELEVENLABS_SDK_URL`), network connections only to this origin and
ElevenLabs, and the microphone only for this origin. The files themselves are owned by
the frontend (see `static/`).
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
VOICE_PAGE = "voice.html"
CDN_SCRIPT_HOST = "https://cdn.jsdelivr.net"
# The only version of the ElevenLabs browser SDK the voice page may load. The CSP allows
# exactly this file (not the whole CDN host); `static/voice.html` must reference the same
# URL (with its SRI hash) and the vendored worklets must come from the same version.
ELEVENLABS_SDK_VERSION = "1.27.0"
ELEVENLABS_SDK_URL = (
    f"{CDN_SCRIPT_HOST}/npm/@elevenlabs/client@{ELEVENLABS_SDK_VERSION}/dist/lib.iife.js"
)
ELEVENLABS_CONNECT_HOSTS = ("wss://api.elevenlabs.io", "https://api.elevenlabs.io")

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'none'",
        f"script-src 'self' {ELEVENLABS_SDK_URL}",
        f"connect-src 'self' {' '.join(ELEVENLABS_CONNECT_HOSTS)}",
        "style-src 'self'",
        "img-src 'self' data:",
        "media-src 'self' blob:",
        "worker-src 'self' blob:",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ]
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class SecureStaticFiles(StaticFiles):
    """`StaticFiles` that adds `SECURITY_HEADERS` to every file it serves."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve `path` as usual, then add the security headers."""
        response = await super().get_response(path, scope)
        response.headers.update(SECURITY_HEADERS)
        return response


def build_router(static_dir: Path) -> APIRouter:
    """Create the page router serving HTML from `static_dir`."""
    router = APIRouter(include_in_schema=False)

    @router.get("/voice")
    def voice_page() -> FileResponse:
        """Serve the voice page (404 if the frontend file is missing)."""
        page = static_dir / VOICE_PAGE
        if not page.is_file():
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(page, media_type="text/html; charset=utf-8", headers=SECURITY_HEADERS)

    return router
