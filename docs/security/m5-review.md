# Security review: M5

Reviewed by security-agent at `e905715` (branch `m5-voice`). **Verdict: PASS** (no Critical or High findings). Findings marked *Fixed* were closed in `28312a7..ee6eaf5` and `d89e428`, tested in `e252aa8`.

| # | Severity | Area | Finding | Status |
|---|----------|------|---------|--------|
| 1 | Medium | Abuse / cost | Without auth, anyone reaching a public deploy could mint signed URLs and drain the ElevenLabs free minutes. | **Fixed**: `VOICE_ENABLED` (off by default in prod), `VOICE_MAX_SESSIONS_PER_DAY` (30, reserved atomically, released on upstream failure). Set a usage limit on the ElevenLabs key. |
| 2 | Low | Abuse / cost | No maximum conversation length. | **Fixed**: `max_duration_seconds` 180. |
| 3 | Low | Agent hardening | Client overrides not explicitly disabled. | **Fixed**: setup script re-asserts overrides on every run. Live-verified: only `first_message` is overridable (for greet-once-per-tab); prompt, LLM, tools and voice stay locked. |
| 4 | Low | Prompt injection to UI | An injected tool call could swap a pending issue preview. | **Fixed**: a pending proposal can't be replaced by the agent; user-initiated replacement shows a notice; Confirm arms after 3 s. |
| 5 | Low | CSP | `script-src` trusted the whole jsDelivr host. | **Fixed**: narrowed to the exact pinned SDK file. |
| 6 | Info | CSP | `blob:` in worker/media-src and the https ElevenLabs origin may be unnecessary. | Open: trim after the live test. |
| 7 | Info | Headers | No HSTS/COOP; JSON responses lack `nosniff`. | Open: M6 deploy. |
| 8 | Info | Client logging | SDK errors could print the signed URL to the console. | **Fixed**: only fixed text and the error name are logged. |
| 9 | Info | Config | `.env.example` placeholder made voice look configured. | **Fixed**: placeholders treated as unset. |
| 10 | Info | Static mount | Vendor README/LICENSE served publicly. | Accepted. |

Passed: API key server-side only (`SecretStr`, pinned host, no redirects, `trust_env=False`, never logged); signed URL validated as `wss://api.elevenlabs.io/v1/convai/...`, `no-store`, never logged; only two client tools (`ask_studiodesk`, `check_bug_report`), no tool can file or confirm; tool results compact (no snippets, tokens or action ids); all text rendered with `textContent`; confirm token in memory only, `isTrusted` clicks; issue links limited to `https://github.com/`; strict CSP, `Permissions-Policy: microphone=(self)`; SDK pinned with SRI, worklets vendored with verified hashes; no static traversal or listing; pip-audit clean.

Live-verified after setup: `enable_auth` true, allowlist `localhost`, exactly two client tools, the only built-in tool is `skip_turn` (silent waiting; cannot file, confirm, transfer or end calls), only the `first_message` override enabled, `max_duration_seconds` 180, `expressive_mode` off, `turn_timeout` 30 s, `gemini-2.0-flash` accepted, signed URLs on `wss://api.elevenlabs.io/v1/convai/`. Still open: trim `blob:` from the CSP if unneeded (#6).
