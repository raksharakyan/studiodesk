# Security review: M4

Reviewed by security-agent at `0a9bdba` (plus the uncommitted live reports). **Verdict: PASS** (no Critical, High or Medium findings). Findings marked *Fixed* were closed in `372fe69..87eb1d8`.

| # | Severity | Area | Finding | Status |
|---|----------|------|---------|--------|
| 1 | Low | Least privilege | `--qdrant cloud` is read-only in code but reuses the app's write-capable Qdrant key. | Documented in `evals/README.md`: use a read-only, collection-scoped key for eval runs. |
| 2 | Low | Report hygiene | The recorded command line could carry absolute local paths into the committed report. | **Fixed**: paths made repo-relative or `<abs>/name`. |
| 3 | Low | Report content | Judge-written failure reasons could repeat a leaked secret into the committed report. | **Fixed**: every report is scanned for configured secrets and token patterns and redacted before writing. |
| 4 | Info | Metric labelling | `no_action_executed` only compared store counts. | **Fixed**: renamed `store_unchanged`; runtime `sys.modules` check plus a clean-interpreter test that no action/GitHub/Slack module loads. |
| 5 | Info | Error output | Non-harness errors exit with a traceback (no secret values). | Open (local only). |
| 6 | Info | Prompt injection | 3 `duplicate` verdicts on injected reports, all genuine matches above the 0.70 gate; no verdict was forced by injected text. | Consistent with M3 #2 (Low); "file anyway" planned for M6. |
| 7 | Info | Cache integrity | Local response cache is not authenticated. | Accepted: gitignored, local only, `cache_hits` recorded in reports. |

Passed: injection strings in cases are only ever passed as user/document data; no real-length secrets in cases, tests or history; cache files hold only `{model, schema, output}` keyed by a hash; `evals/.cache/` and offline reports gitignored; committed reports contain no secrets, cluster URLs or local paths; judge prompt delimits answer and sources and the harness computes groundedness itself from source ids actually shown; CI offline step uses no secrets or network; pip-audit clean.
