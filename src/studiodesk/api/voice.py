"""`GET /voice/session`: mint a signed ElevenLabs conversation URL for the browser.

The ElevenLabs API key stays server-side; the browser only receives a signed URL that is
valid for starting one session within 15 minutes. That URL is a bearer credential, so the
response is `Cache-Control: no-store` and the URL is never logged. Upstream failures
return a generic 502; a missing key or agent id returns 503 "Voice is not configured".
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from slowapi import Limiter

from studiodesk.api.deps import get_voice, get_voice_agent_id
from studiodesk.models.voice import VoiceSessionResponse
from studiodesk.voice import SIGNED_URL_TTL_S, ElevenLabsClient, ElevenLabsError

logger = logging.getLogger(__name__)

VOICE_UPSTREAM_DETAIL = "Voice is temporarily unavailable"
NO_STORE = {"Cache-Control": "no-store"}


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the voice router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["voice"])

    @router.get("/voice/session", response_model=VoiceSessionResponse)
    @limiter.limit(rate_limit)
    def voice_session(
        request: Request,
        response: Response,
        voice: Annotated[ElevenLabsClient, Depends(get_voice)],
        agent_id: Annotated[str, Depends(get_voice_agent_id)],
    ) -> VoiceSessionResponse:
        """Return a fresh signed URL for one voice conversation (never cached or logged)."""
        response.headers.update(NO_STORE)
        try:
            signed_url = voice.get_signed_url(agent_id)
        except ElevenLabsError as exc:
            # str(exc) carries only the exception class or status, never the URL or key.
            logger.warning("voice session failed", extra={"error": str(exc)})
            raise HTTPException(
                status_code=502, detail=VOICE_UPSTREAM_DETAIL, headers=NO_STORE
            ) from None
        return VoiceSessionResponse(signed_url=signed_url, expires_in_s=SIGNED_URL_TTL_S)

    return router
