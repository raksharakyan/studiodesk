"""Slack alert after an issue is filed, via an incoming webhook on an injected client.

With no `SLACK_WEBHOOK_URL` configured every alert is a no-op that logs
`slack alert skipped (not configured)`. Messages are plain text (`mrkdwn: false`) with
`&`, `<` and `>` escaped, so `<!channel>`, `<@U123>` and link syntax are inert, and
`@channel`/`@here`/`@user` get a zero-width joiner. The webhook URL is a secret: it is
never logged, and failures report only the exception class or HTTP status.
"""

import logging

import httpx
from pydantic import SecretStr

from studiodesk.actions.issue import neutralise_mentions
from studiodesk.models.actions import CreatedIssue, IssueDraft, SlackStatus

logger = logging.getLogger(__name__)

SKIPPED_MESSAGE = "slack alert skipped (not configured)"


def escape_slack_text(text: str) -> str:
    """Escape Slack control characters and neutralise mentions."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return neutralise_mentions(escaped)


def issue_alert_text(issue: CreatedIssue, draft: IssueDraft) -> str:
    """Plain-text alert for a newly filed issue."""
    labels = ", ".join(draft.labels) or "none"
    return escape_slack_text(
        f"New bug filed via StudioDesk: #{issue.number} {draft.title}\n"
        f"Labels: {labels}\n{issue.url}"
    )


class SlackNotifier:
    """Posts alerts to one webhook, or does nothing when none is configured."""

    def __init__(self, http: httpx.Client, webhook_url: SecretStr | None, timeout_s: float) -> None:
        """Use `http` (owned by the caller) to post to `webhook_url` (None = disabled)."""
        self._http = http
        self._webhook_url = webhook_url
        self._timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        """True if a webhook is configured."""
        return self._webhook_url is not None

    def notify_issue_created(self, issue: CreatedIssue, draft: IssueDraft) -> SlackStatus:
        """Send the alert; returns sent, skipped (not configured) or failed. Never raises."""
        if self._webhook_url is None:
            logger.info(SKIPPED_MESSAGE)
            return SlackStatus.SKIPPED
        try:
            response = self._http.post(
                self._webhook_url.get_secret_value(),
                json={"text": issue_alert_text(issue, draft), "mrkdwn": False},
                timeout=self._timeout_s,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            logger.warning("slack alert failed", extra={"error": type(exc).__name__})
            return SlackStatus.FAILED
        if response.status_code != httpx.codes.OK:
            logger.warning("slack alert failed", extra={"status": response.status_code})
            return SlackStatus.FAILED
        logger.info("slack alert sent")
        return SlackStatus.SENT
