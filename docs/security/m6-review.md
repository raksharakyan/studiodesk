# Security review: M6

Two reviews by security-agent on branch `m6-web-deploy`: the public-deploy review (at `341265f`, for a live Hugging Face Space) and the showcase review (at `0ebc38a`). **Verdict: PASS** for both (no Critical or High findings).

Hugging Face now requires a PRO subscription for Docker Spaces, so the published deliverable is a **static GitHub Pages showcase** that replays results recorded from the real pipeline. No backend is publicly reachable; the live-deploy findings below apply only to a future live deployment (`scripts/deploy_hf_space.py` with HF PRO, or another host).

## Public-deploy review (live backend)

| # | Severity | Area | Finding | Status |
|---|----------|------|---------|--------|
| 1 | Medium | LLM quota / abuse | No global LLM budget; one visitor could exhaust Groq's free 8k tokens/min. | **Fixed**: process-wide budget (6 calls/min, 300/day), prod per-IP limits (`ask 3/minute;30/day`, `bugs 3/minute;20/day`, `actions 2/minute;5/day`, `voice 2/minute;5/day`). Applies to live deploys only. |
| 2 | Medium | Proxy / client IP | `TRUSTED_PROXY_IPS` unknown on the host; unset means one shared bucket, too broad allows spoofing. | **Mitigated**: `DEBUG_LOG_CLIENT_IP` to read the proxy chain; deploy script rejects public proxy ranges. Determine empirically at a live deploy. |
| 3 | Low | Daily caps | Global in-memory caps reset on restart. | Accepted; provider-side limits (ElevenLabs credit cap, Groq free tier) are the backstop. |
| 4 | Low | Outbound content | Public sandbox repo can receive spam (5/day). | Accepted; fenced content, broken mentions/refs, allowlisted labels. |
| 5 | Low | Deploy staging | Untracked files under `src/`/`data/` could be uploaded. | **Fixed**: only `git ls-files` are staged; untracked files abort. |
| 6 | Low | Secret scan | Missed encoded secrets, token shapes and quoted `.env` values. | **Fixed**: python-dotenv parsing, token patterns, base64/URL/JSON-escaped forms. |
| 7 | Low | Rate limiting | IPv6 per-address keys bypassable within a /64. | **Fixed**: IPv6 keyed by /64. |
| 8–13 | Info | Mirror upload, settings split, frame-ancestors, logs, license, CSP `blob:` | | Frame-ancestors limited to `https://huggingface.co`; MIT `LICENSE` added; others accepted or open (CSP `blob:` trim). |

## Showcase review (published)

| # | Severity | Area | Finding | Status |
|---|----------|------|---------|--------|
| 1 | Low | CI least privilege | `pages: write` / `id-token: write` granted to the build job, which runs project code. | **Fixed**: workflow is `contents: read`; only the deploy job (pinned GitHub actions, no project code) can write to Pages. |
| 2 | Low | Secret scan | CI build couldn't detect a Qdrant cluster URL (no `.env` in CI). | **Fixed**: cluster-host pattern (placeholder excluded), with tests. |
| 3 | Info | `showcase.json` | No tokens, action ids, internal hosts or PII; injected text only as fenced data; off-topic declined. | Safe to publish. |
| 4 | Info | Site HTML/JS | CSP meta, `textContent` only, strict link validation, single same-origin fetch. | OK. |
| 5 | Info | Pages workflow | Push to `main` and manual only (no fork PRs), SHA-pinned actions, no secrets, artifact is `_site` only. | OK. |
| 6 | Info | Recorder | Can't file issues or contact GitHub/Slack/ElevenLabs; only `/ask` and `/bugs/check`; all output scanned before writing. | OK. |

Closed earlier items in M6: M2 #3 (actions pinned by SHA), #4 (base images pinned by digest), #8 (422 without input echo), #10 (image built and smoke-tested in CI); M3 #2 ("file anyway" control); M5 #7 (HSTS/COOP/nosniff). Still open (Info): M2 #6 (HF cache re-hash), M2 #11 (dummy `sk-ant-` strings in tests), M4 #5 (local traceback), M5 #6 (CSP `blob:` trim).
