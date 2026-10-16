"""HTML pages and static assets, served with strict security headers.

`GET /` serves `static/index.html` (the main page); `GET /voice` serves `static/voice.html`
or, when the frontend dropped that file, redirects to the main page's voice section
(`/#voice`). `/static` serves the rest of `static/`. All of them carry
a Content-Security-Policy that allows scripts only from this origin and the one pinned SDK
file on the CDN (`ELEVENLABS_SDK_URL`), network connections only to this origin and
ElevenLabs, and the microphone only for this origin. The files themselves are owned by
the frontend (see `static/`).
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_PAGE = "index.html"
VOICE_PAGE = "voice.html"
VOICE_SECTION_URL = "/#voice"
CDN_SCRIPT_HOST = "https://cdn.jsdelivr.net"
# The only version of the ElevenLabs browser SDK the voice page may load. The CSP allows
# exactly this file (not the whole CDN host); `static/voice.html` must reference the same
# URL (with its SRI hash) and the vendored worklets must come from the same version.
ELEVENLABS_SDK_VERSION = "1.27.0"
ELEVENLABS_SDK_URL = (
    f"{CDN_SCRIPT_HOST}/npm/@elevenlabs/client@{ELEVENLABS_SDK_VERSION}/dist/lib.iife.js"
)
ELEVENLABS_CONNECT_HOSTS = ("wss://api.elevenlabs.io", "https://api.elevenlabs.io")
# The only site allowed to frame the pages: the Hugging Face Space page embeds the app
# (served from *.hf.space) in an iframe. No X-Frame-Options is sent (it cannot allowlist
# an origin); Cross-Origin-Opener-Policy only applies to top-level documents, so it does
# not affect the embed.
FRAME_ANCESTOR = "https://huggingface.co"

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
        f"frame-ancestors {FRAME_ANCESTOR}",
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


def html_page(page: Path) -> FileResponse:
    """Serve `page` as HTML with `SECURITY_HEADERS` (404 if the file is missing)."""
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(page, media_type="text/html; charset=utf-8", headers=SECURITY_HEADERS)


def build_router(static_dir: Path) -> APIRouter:
    """Create the page router serving HTML from `static_dir`."""
    router = APIRouter(include_in_schema=False)

    @router.get("/")
    def index_page() -> FileResponse:
        """Serve the main page (404 if the frontend file is missing)."""
        return html_page(static_dir / INDEX_PAGE)

    @router.get("/voice", response_model=None)
    def voice_page() -> FileResponse | RedirectResponse:
        """Serve the voice page; without it, redirect to `/#voice` if the main page exists."""
        voice = static_dir / VOICE_PAGE
        if not voice.is_file() and (static_dir / INDEX_PAGE).is_file():
            return RedirectResponse(VOICE_SECTION_URL, status_code=307, headers=SECURITY_HEADERS)
        return html_page(voice)

    return router
