# Security

StudioDesk is a portfolio project. Its public deliverable is a static GitHub Pages showcase with no backend. The full app is built to be run locally or as a single-process deployment. This document covers the threat model, the controls in place, the per-milestone reviews and what is still open. Review details are in [`docs/security/`](docs/security/).

## Scope and threat model

**Assets**
- API keys: Groq or Anthropic, Qdrant, the GitHub PAT, ElevenLabs, the Slack webhook, and `HF_TOKEN` (used only on the owner's machine).
- The sandbox GitHub repository that issues are filed into (`GITHUB_REPO`).
- Provider quotas and money: Groq free-tier tokens, ElevenLabs credits, and the daily issue and voice caps.
- Users' trust in answers: citations must point at documents that were actually retrieved, and verdicts must not be forced by injected text.

**Adversaries**
- Anonymous visitors to a public deployment. There is no auth by design, so anyone who can reach the API is in scope.
- Poisoned documents in the corpus. The dataset includes planted injection fixtures (BUG-0014, BUG-0026, BUG-0041, BUG-0054 and a "Community tips" section in `data/synthetic/docs/troubleshooting-guide.md`).
- Malicious bug reports submitted to `/bugs/check`, aimed at the LLM, the issue body, mentions or labels.
- Compromised dependencies, actions or images in the supply chain.

**Trust boundaries**

| Boundary | Trusted side | Untrusted input crossing it |
|---|---|---|
| Browser → API | API | All request fields, headers, `X-Forwarded-For` |
| API → LLM provider | API | Model output (answers, reasons, cited ids) |
| API → Qdrant | API | Stored chunk text, which may be poisoned |
| API → GitHub / Slack | Remote services | No model text; only server-built content goes out (a model verdict can at most add the allowlisted `possible-duplicate` label and our own ids) |
| Browser ⇄ ElevenLabs | API (holds the key) | Transcripts and tool-call arguments chosen by the voice agent |

## Controls

### Secrets
- All keys are `SecretStr` in `studiodesk.config.Settings`, read from the environment or `.env`. `.env` is ignored by git and Docker, and `.env.example` holds placeholders that count as unset.
- Log redaction covers messages, extras, tracebacks and JSON-escaped secrets. uvicorn logs go through the same handler. `hide_input_in_errors` is set, and errors carry only class names or HTTP statuses.
- The GitHub, Slack and ElevenLabs clients use `trust_env=False`, and the Groq and Anthropic SDK clients pin their base URL so env overrides cannot redirect them. `qdrant_url` rejects userinfo, query and fragment, and must be HTTPS except on localhost.
- The ElevenLabs key never reaches the browser. Signed URLs are returned with `Cache-Control: no-store` and never logged.
- `secretscan.py` checks every published artifact (Space upload, `showcase.json`, `_site/`). It looks for known token shapes, configured secret values in raw, base64, URL- and JSON-encoded forms, and Qdrant cluster hosts.

### Prompt injection
- `prompts.py` places retrieved chunks and user text only inside `<document>`, `<user_question>` or `<new_bug_report>` tags and neutralises our tag names inside them. System prompts are frozen constants.
- Citation filtering is enforced on the server: ids that were not retrieved are removed from sources and answer text (`agent/answer.py`). The model is not trusted to cite correctly.
- No model text reaches an outward action. Issue titles, bodies and labels are built from validated user fields, routing enums and our own ids (`actions/issue.py`).
- A `duplicate` verdict (which withholds the proposal) needs a score ≥ 0.70, above the highest hard-negative score of 0.684 (`agent/duplicates.py`). The UI offers "file anyway".
- Eval coverage: 14 injection cases, all 14 passing ([`evals/reports/latest.md`](evals/reports/latest.md)). They check for no secrets, no system-prompt leak, only retrieved ids, no forced verdict and an unchanged store. The harness never imports the GitHub, Slack or proposal modules, which a clean-interpreter test verifies.

### Outward actions (GitHub, Slack)
- **Propose, then confirm.** `/bugs/check` stores the exact draft. `POST /actions/{id}/confirm` runs the stored draft and takes only the id and the token from the request.
- **Token design.** Tokens are `secrets.token_urlsafe(32)` (256-bit). Only the SHA-256 digest is stored, it is compared with `hmac.compare_digest`, and claiming is atomic and tested under concurrency. Tokens are single-use with a 15-minute TTL (`ACTION_TTL_S=900`). A wrong token or unknown id returns 404, a replay 409 and an expired token 410.
- **Caps and kill switches.** `ACTIONS_ENABLED` is off by default in prod. `ACTIONS_MAX_PER_DAY=20` is reserved at claim time and freed if creation fails.
- **Content.** User fields go inside fences longer than any backtick run they contain. `@mentions` and `#N` / `owner/repo#N` references in titles are broken with a zero-width joiner. Labels come from a fixed allowlist (`actions/labels.py`, created by `scripts/setup_issue_labels.py`).
- **Transport.** Hosts are pinned (`api.github.com`, `hooks.slack.com`, `api.groq.com`, `api.anthropic.com`, `api.elevenlabs.io`). The GitHub, Slack and ElevenLabs clients do not follow redirects. Every call has a timeout, and the returned `html_url` is checked exactly. Slack messages are plain text with `<`, `>` and `&` escaped and mentions neutralised.
- **Credentials.** The GitHub token is a fine-grained PAT for one repository with Issues read/write only.

### Voice
- Only two client tools exist (`ask_studiodesk`, `check_bug_report`) and there is no confirm tool. The only built-in tool is `skip_turn`. The agent cannot replace a pending proposal, and Confirm arms after 3 s and accepts only `isTrusted` clicks.
- The agent is private (`enable_auth` with a host allowlist). Only `first_message` can be overridden by the client. Calls are capped at 180 s (`voice/agent.json`). `VOICE_ENABLED` is off in prod, and `VOICE_MAX_SESSIONS_PER_DAY=30` (per process; it resets on restart) reserves a slot before ElevenLabs is called.

### Abuse and cost
- Per-IP rate limits via slowapi. IPv6 is keyed by /64. `X-Forwarded-For` is honoured only from `TRUSTED_PROXY_IPS` and resolved right to left. Prod defaults are `ask 3/minute;30/day`, `bugs 3/minute;20/day`, `actions 2/minute;5/day` and `voice 2/minute;5/day`.
- A global LLM budget across all clients (`LLM_MAX_CALLS_PER_MINUTE=6`, `LLM_MAX_CALLS_PER_DAY=300`). Concurrency gate of 4, a 30 s timeout and a 75 s per-call deadline. When exhausted, the API returns 503 with `Retry-After`.
- Request size: `MAX_REQUEST_BYTES=64000`, enforced on `Content-Length` and on streamed bodies (413). Pydantic field limits apply on top, for example a question of at most 1,000 characters.

### Web
- A strict CSP allows scripts only from self plus the single pinned ElevenLabs SDK file, which has an SRI `sha384` hash. The audio worklets are vendored with verified hashes.
- Headers: HSTS and COOP in prod, `nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy: microphone=(self)`, and `frame-ancestors` limited to `https://huggingface.co`.
- All model and corpus text is rendered with `textContent`, issue links are limited to `https://github.com/`, and the confirm token is kept in memory only.
- 422 responses do not echo user input. OpenAPI docs are disabled in prod.

### Supply chain
- `uv.lock` is committed and CI runs `uv sync --frozen`. `pip-audit` blocks the build, and a second audit covers the public torch version (pip-audit skips `+cpu` builds).
- GitHub Actions are pinned by full commit SHA, `actions/checkout` runs with `persist-credentials: false`, and both Docker base images are pinned by digest.
- The embedding model is pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` with `trust_remote_code=False` and is downloaded only at build time. On Linux, torch comes from an explicit CPU-only index (`pyproject.toml`), which prevents dependency confusion.
- The container runs as non-root uid 1000 with root-owned app files (`Dockerfile`), and the CI smoke test checks the uid.

### Deploy and showcase
- `scripts/deploy_hf_space.py` stages only `git ls-files`, aborts on untracked files, secret-scans the upload, never copies `HF_TOKEN` and rejects public proxy ranges for `TRUSTED_PROXY_IPS` unless `--force-public-proxy` is passed.
- `scripts/record_showcase.py` calls only `/ask` and `/bugs/check`. GitHub and Slack are replaced by stand-ins that raise if called, and tokens and action ids are nulled before writing.
- `.github/workflows/pages.yml` runs only on pushes to `main` and manual dispatch, never on fork PRs. The build job is `contents: read`; only the deploy job, which runs pinned GitHub actions and no project code, gets `pages: write` and `id-token: write`.

## Review summary

Each milestone was reviewed by security-agent, and High or Critical findings block a merge. Counts below are findings as raised, including those fixed afterwards.

| Milestone | Verdict | Critical | High | Medium | Low | Info | Review |
|---|---|---|---|---|---|---|---|
| M1 foundation | PASS | 0 | 0 | 0 | 7 | 5 | [m1-review](docs/security/m1-review.md) |
| M2 ingest + search | PASS | 0 | 0 | 1 | 4 | 6 | [m2-review](docs/security/m2-review.md) |
| M3 agent + actions | PASS | 0 | 0 | 1 | 3 | 4 | [m3-review](docs/security/m3-review.md) |
| M4 evals | PASS | 0 | 0 | 0 | 3 | 4 | [m4-review](docs/security/m4-review.md) |
| M5 voice | PASS | 0 | 0 | 1 | 4 | 5 | [m5-review](docs/security/m5-review.md) |
| M6 public deploy | PASS | 0 | 0 | 2 | 5 | 6 | [m6-review](docs/security/m6-review.md) |
| M6 showcase | PASS | 0 | 0 | 0 | 2 | 4 | [m6-review](docs/security/m6-review.md) |

All Medium findings are fixed or mitigated. M6 #2 (proxy IP) is mitigated and can only be closed during a live deploy.

## Known open items

- **CSP `blob:`** in `media-src` and `worker-src` may be unnecessary and should be trimmed after a live voice test (M5 #6, Info).
- **HF model cache is not re-hashed after a CI cache restore.** This is low risk: safetensors, no remote code (M2 #6, Info).
- **Dummy key-like strings** (`sk-ant-...`) in tests may trip secret scanners (M1 #11, M2 #11, Info).
- **Local tracebacks.** Non-harness errors in local CLIs exit with a traceback, which contains no secret values (M4 #5, Info).
- **Eval runs against Qdrant Cloud** reuse the app's write-capable key unless a read-only, collection-scoped key is configured (M4 #1, Low, documented).
- **Live-deploy items.** These do not affect the static showcase:
  - `TRUSTED_PROXY_IPS` must be determined empirically on the host with `DEBUG_LOG_CLIENT_IP` (M6 #2, Medium, mitigated).
  - Proposals, rate limits, the LLM budget and voice quotas are kept per process. With N workers the caps multiply and a confirm works only on the process that created the proposal (M3 #8).
  - A restart resets the in-memory daily caps. The backstop is provider-side limits: an ElevenLabs credit cap and the Groq free tier (M6 #3, Low, accepted).
  - The public sandbox repo can receive spam issues, bounded by the prod limit of `actions 2/minute;5/day` per IP and `ACTIONS_MAX_PER_DAY=20` per process (M6 #4, Low, accepted).
- **No auth by design.** A real deployment needs auth and per-user quotas.
- **LLM SDK clients.** The Groq/Anthropic SDK clients use their libraries' default redirect and proxy-environment behaviour (hosts are pinned); hardening them like the GitHub/ElevenLabs clients is a next step.
- **Accepted risks** (see the reviews): embedded local Qdrant persists with pickle, so `.qdrant_data/` must be trusted; prod requires `QDRANT_URL` (M2 #9, Info). The local eval cache is unauthenticated (gitignored; M4 #7, Info). The vendored SDK README/LICENSE are served publicly (M5 #10, Info). The other M6 deploy Info items (#8–13) are listed in the [M6 review](docs/security/m6-review.md).

## Reporting a vulnerability

Please report vulnerabilities privately through a GitHub security advisory: [open a new advisory](https://github.com/raksharakyan/studiodesk/security/advisories/new) on the repository's Security tab. Include the affected endpoint or file, the steps to reproduce and the impact. Please do not open a public issue for security problems.
