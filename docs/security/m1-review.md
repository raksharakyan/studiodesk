# Security review: M1

Reviewed by security-agent at `f677f61`. **Verdict: PASS** (no Critical or High findings).

| # | Severity | Area | Finding | Plan |
|---|----------|------|---------|------|
| 1 | Low | Input validation | `max_request_bytes` is configured but not enforced (M1 has no input endpoints). | Enforce with 413 middleware + field `max_length` in M2. |
| 2 | Low | CI | Actions pinned to major tags, not SHAs. | Pin SHAs + Dependabot. |
| 3 | Low | CI | `pip-audit` is advisory (`continue-on-error`). | Make blocking in M2. |
| 4 | Low | CI | `actions/checkout` persists credentials. | `persist-credentials: false`. |
| 5 | Low | Container | Base images use floating tags. | Pin digests. |
| 6 | Low | Container | `.venv`/`data` owned by the runtime user. | Root-owned, read-only. |
| 7 | Low | Logging | uvicorn loggers bypass the redacting JSON handler. | Route uvicorn logs through it before M2/M3. |
| 8 | Info | Config | No scheme/host checks on Qdrant and Slack URLs. | Require HTTPS; Slack host `hooks.slack.com`. |
| 9 | Info | Errors | Loader errors include input values (fine for CLI). | Generic errors if exposed via API. |
| 10 | Info | Loader | No file size cap; trusted paths only. | Size cap + base-dir check if user uploads are added. |
| 11 | Info | Secrets | Fake `sk-ant-` test strings may trip scanners. | Use non-matching dummies or allowlist. |
| 12 | Info | Prompt injection | BUG-0014/0026/0041/0054 are inert fixtures today. | Delimit retrieved docs, confirm-gated actions, escape in UI, eval cases (M2–M4). |

Passed: no secrets in tree or history; `.env` ignored by git and Docker; all keys are `SecretStr`; log redaction covers messages, extras, tracebacks and JSON-escaped secrets; docs/openapi disabled in prod; non-root slim container; CI `contents: read`; pip-audit found no known vulnerabilities (62 packages incl. dev).
