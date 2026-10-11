"""Models for `GET /voice/session`."""

from pydantic import BaseModel, ConfigDict, Field


class VoiceSessionResponse(BaseModel):
    """A short-lived signed URL the browser uses to start one ElevenLabs conversation.

    `signed_url` is a bearer credential for that conversation: it is returned with
    `Cache-Control: no-store` and never logged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    signed_url: str = Field(min_length=1, max_length=4096)
    expires_in_s: int = Field(gt=0)
