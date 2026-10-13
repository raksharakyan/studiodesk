"""`GET /voice/session`: mint a signed ElevenLabs conversation URL for the browser.

The ElevenLabs API key stays server-side; the browser only receives a signed URL that is
valid for starting one session within 15 minutes. That URL is a bearer credential, so the
response is `Cache-Control: no-store` and the URL is never logged.

Responses besides 200: 503 "Voice is disabled" when VOICE_ENABLED is off (the default in
prod); 503 "Voice is not configured" without a key or agent id; 503 "Daily voice limit
reached" with `Retry-After` (seconds until UTC midnight) once VOICE_MAX_SESSIONS_PER_DAY
sessions were minted today; a generic 502 on upstream failures. A daily slot is reserved
before ElevenLabs is called and given back if the call fails.
"""

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from slowapi import Limiter

from studiodesk.api.deps import (
    VOICE_UNAVAILABLE_DETAIL,
    get_voice,
    get_voice_agent_id,
    get_voice_quota,
    require_voice_enabled,
)
from studiodesk.models.voice import VoiceSessionResponse
from studiodesk.voice import (
    SIGNED_URL_TTL_S,
    ElevenLabsClient,
    ElevenLabsError,
    VoiceDailyLimitError,
    VoiceSessionQuota,
)

logger = logging.getLogger(__name__)

VOICE_UPSTREAM_DETAIL = VOICE_UNAVAILABLE_DETAIL
VOICE_DAILY_LIMIT_DETAIL = "Daily voice limit reached"
NO_STORE = {"Cache-Control": "no-store"}


def build_router(limiter: Limiter, rate_limit: str) -> APIRouter:
    """Create the voice router, rate-limited per client IP by `limiter` at `rate_limit`."""
    router = APIRouter(tags=["voice"])

    @router.get(
        "/voice/session",
        response_model=VoiceSessionResponse,
        # Runs before the other dependencies: a disabled voice says so first.
        dependencies=[Depends(require_voice_enabled)],
    )
    @limiter.limit(rate_limit)
    def voice_session(
        request: Request,
        response: Response,
        voice: Annotated[ElevenLabsClient, Depends(get_voice)],
        agent_id: Annotated[str, Depends(get_voice_agent_id)],
        quota: Annotated[VoiceSessionQuota, Depends(get_voice_quota)],
    ) -> VoiceSessionResponse:
        """Return a fresh signed URL for one voice conversation (never cached or logged)."""
        response.headers.update(NO_STORE)
        day = _reserve_slot(quota)
        try:
            signed_url = voice.get_signed_url(agent_id)
        except ElevenLabsError as exc:
            quota.release(day)
            # str(exc) carries only the exception class or status, never the URL or key.
            logger.warning("voice session failed", extra={"error": str(exc)})
            raise HTTPException(
                status_code=502, detail=VOICE_UPSTREAM_DETAIL, headers=NO_STORE
            ) from None
        except BaseException:
            quota.release(day)  # unexpected failure: no session was minted either
            raise
        return VoiceSessionResponse(signed_url=signed_url, expires_in_s=SIGNED_URL_TTL_S)

    return router


def _reserve_slot(quota: VoiceSessionQuota) -> date:
    """Reserve today's slot or raise 503 with Retry-After until UTC midnight."""
    try:
        return quota.reserve()
    except VoiceDailyLimitError as exc:
        logger.warning("daily voice session limit reached")
        raise HTTPException(
            status_code=503,
            detail=VOICE_DAILY_LIMIT_DETAIL,
            headers={**NO_STORE, "Retry-After": str(exc.retry_after_s)},
        ) from None
