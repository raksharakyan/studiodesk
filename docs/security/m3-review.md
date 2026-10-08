# Security review: M3

Reviewed by security-agent at `d2f9b99`. **Verdict: PASS** (no Critical or High findings). Findings marked *Fixed* were closed in `11e4485..e7c72bb`.

| # | Severity | Area | Finding | Status |
|---|----------|------|---------|--------|
| 1 | Medium | Abuse (no auth by design) | Anyone reaching a public deployment could check a bug and confirm their own proposal, filing issues as the token owner. | **Fixed**: `ACTIONS_ENABLED` kill switch (off by default in prod) and `ACTIONS_MAX_PER_DAY` cap (20), reserved atomically at claim time. |
| 2 | Low | Prompt injection / integrity | An injected corpus report could push the LLM toward "duplicate"; a `duplicate` verdict withholds the proposal. Bounded by the 0.70 score gate (above every hard negative) and owner-only ingestion. | Open: M4 eval cases; "file anyway" control in the M6 UI. |
| 3 | Low | Outbound content | User markdown in issue bodies could create cross-repo references or render differently from the preview. | **Fixed**: user fields fenced with an unescapable fence; `#N` / `owner/repo#N` and `@` broken in titles. |
| 4 | Low | DoS / cost | One slow LLM call could hold a worker for minutes; no global concurrency limit. | **Fixed**: 30 s timeout, global concurrency gate (4, 503 + `Retry-After` when busy), 75 s per-request deadline. |
| 5 | Info | Output handling | Answers, candidate reasons, titles and snippets are model or corpus text. | Open: M6 UI renders as text only. |
| 6 | Info | Config | `.env.example` Slack placeholder passed validation. | **Fixed** |
| 7 | Info | Outbound HTTP | Shared client trusted proxy/netrc environment. | **Fixed**: `trust_env=False` |
| 8 | Info | Proposal store | Per-process, in memory. | Documented; single-process deploy, shared store needed to scale out. |

Passed: delimited prompts with tag neutralisation and frozen system prompts; issue title/body/labels built server-side from validated fields, routing enums and our own ids (no model text reaches outward actions); citations filtered to the retrieved set and unretrieved ids stripped from answer text; 256-bit single-use confirm tokens stored as SHA-256, constant-time compare, atomic claim (tested under concurrency), 15-minute TTL, stored-draft execution; GitHub pinned to `api.github.com` with no redirects and exact `html_url` verification; Slack host-pinned, escaped, mentions neutralised; Groq/Anthropic hosts pinned against env overrides; per-route rate limits; secrets `SecretStr` and redacted, `hide_input_in_errors`; no secrets in history; fine-grained PAT limited to one repo with Issues read/write; pip-audit clean (anthropic, groq, httpx2 added).

Live verification (orchestrator): Groq `openai/gpt-oss-120b` answered with valid citations; an injection probe retrieving BUG-0026 produced a normal answer with no leak and no action; confirm filed `raksharakyan/starfall-outpost-issues#1`; replay → 409, wrong token → 404.

Still open from M2: SHA-pinned actions (#3), digest-pinned base images (#4), HF cache re-hash (#6), 422 input echo (#8, UI must render as text), Docker build in CI (#10), token-like test strings (#11).

Notes for M4–M6: injection eval suite (BUG-0014/0026/0041/0054 as retrieved docs and duplicate candidates; no secrets, no system prompt, no actions, verdict not pushed to duplicate); UI renders all model/corpus text with `textContent`, keeps the confirm token in memory only, same-origin with a CSP, adds "file anyway"; deploy as a single process with `TRUSTED_PROXY_IPS`, provider spend caps and a PAT expiry.
