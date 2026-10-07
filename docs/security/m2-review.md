# Security review: M2

Reviewed by security-agent at `a280d3d`. **Verdict: PASS** (no Critical or High findings). Findings marked *Fixed* were closed in `a356ec1..96fcb4f` and covered by tests in `eb8ec53`.

| # | Severity | Area | Finding | Status |
|---|----------|------|---------|--------|
| 1 | Medium | Rate limiting | Behind a platform proxy every client shared one bucket; trusting `X-Forwarded-For` naively allows spoofing. | **Fixed**: `TRUSTED_PROXY_IPS` + right-to-left XFF resolution. In-memory limiter is per process (documented). |
| 2 | Low | Supply chain | `torch==x+cpu` is skipped by pip-audit on Linux CI. | **Fixed**: CI audits the public torch version (blocking). |
| 3 | Low | CI | Actions pinned to tags, not SHAs. | Open |
| 4 | Low | Container | Base images on floating tags; builder/runtime distro may drift. | Open |
| 5 | Low | Config | `qdrant_url` accepted userinfo/query/fragment (credentials could bypass redaction). | **Fixed** |
| 6 | Info | CI cache | HF cache not re-hashed after restore. | Open (low risk: safetensors, no remote code) |
| 7 | Info | Input validation | Non-ASCII digits in `Content-Length` raised a 500 via direct ASGI. | **Fixed** |
| 8 | Info | Errors | 422 responses echo user input. | Open: UI must render as text (M6) |
| 9 | Info | Local mode | Embedded Qdrant persists with pickle; `.qdrant_data` must be trusted. | Mitigated: gitignored, prod requires `QDRANT_URL` |
| 10 | Info | Container | Docker image not built in CI. | Open: smoke test at M6 deploy |
| 11 | Info | Secrets hygiene | `sk-ant-` dummy strings in tests may trip scanners. | Open |

Passed: Qdrant key is `SecretStr` end to end and never logged (httpx quieted, errors carry class names only, `hide_input_in_errors`); HTTPS enforced with timeouts and bounded retries; `/search` uses enum-only filters, bounded sizes, generic 503/429/413; prompt-injection fixtures stored and returned as inert data; embedding model pinned by SHA with `trust_remote_code=False`, downloaded only at build time; explicit CPU torch index (no dependency confusion); root-owned app files, non-root user; pip-audit clean.

Notes for M3: delimit retrieved chunks with doc ids and neutralise closing delimiters; propose-then-confirm actions with single-use server-side tokens; duplicate verdicts are advisory only; GitHub/Slack calls to fixed hosts with timeouts and no redirects; neutralise Slack mentions; stricter limits on LLM and action endpoints; never log `ValidationError.errors()` without `include_input=False`.
