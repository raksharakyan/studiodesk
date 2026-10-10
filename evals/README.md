# StudioDesk evals

Labelled cases, a single-command harness and a markdown + JSON report for retrieval,
cited answers, duplicate detection, routing and prompt-injection resistance.

## Run

```bash
uv run python -m evals.run                    # live: real LLM + judge, real MiniLM, fresh in-memory Qdrant
uv run python -m evals.run --qdrant cloud     # search the QDRANT_URL collection read-only (no ingest)
uv run python -m evals.run --offline          # no keys/network: fake LLM + hashing embedder (CI smoke)
uv run python -m evals.run --suite answer --suite injection   # subset (repeatable)
```

| Option | Default | Meaning |
|---|---|---|
| `--suite` | all | `retrieval`, `answer`, `duplicates`, `routing`, `injection` |
| `--offline` | off | fake LLM and embedder; writes `reports/offline.*` (gitignored); scores are meaningless |
| `--qdrant` | `memory` | `cloud` = read-only search of the configured collection |
| `--max-rpm` / `--max-tpm` | `EVAL_MAX_RPM` / `EVAL_MAX_TPM` | per-model request / token budget per 60 s |
| `--no-cache` | off | ignore cached outputs (fresh results are still written to the cache) |
| `--judge-model` | `EVAL_JUDGE_MODEL` | groundedness judge (same provider) |
| `--strict` | off | exit 1 if a metric is below `thresholds.json` (rejected with `--offline`) |
| `--thresholds` | `evals/thresholds.json` | floors file |
| `--limit` | none | max cases per suite (smoke runs) |

Settings come from the environment / `.env` through the app's `Settings` plus these
eval-only variables:

| Variable | Default | Notes |
|---|---|---|
| `EVAL_JUDGE_MODEL` | `qwen/qwen3.8-27b` on Groq; the answerer model on Anthropic | a different model family from the answerer reduces self-grading bias |
| `EVAL_MAX_RPM` | `20` | requests per minute, per model |
| `EVAL_MAX_TPM` | `6000` | tokens per minute, per model (Groq free tier: 8,000) |

## Exit codes

- `0`: every case completed (metrics may still be low unless `--strict`).
- `1`: at least one case **errored** (LLM or vector-store call raised), or `--strict` and a
  completed metric is below its floor.
- `2`: configuration error (missing key or `QDRANT_URL`, invalid case or thresholds file).

CI runs the offline smoke through pytest and must not use `--strict`.

## Rate limits and cost

Groq's free tier allows about 30 requests and **8,000 tokens per minute per model**; the
token limit binds first. Each model gets a sliding 60 s window kept under both budgets.
An uncached call is charged `chars(system + user + schema) / 4` plus the expected output
(the request's `LLM_MAX_TOKENS` until real usage is seen, then the largest observed
output) and corrected with the usage the adapter reports. A cold full run is about 100
calls; at the ~1,600 tokens per call observed on the first live run, that is about 3-4
calls per minute per model, so roughly 20-30 minutes on the free tier; re-runs hit the cache in `evals/.cache/`
(model outputs only, never prompts or keys; gitignored, safe to delete).

## Cases (`cases/*.jsonl`)

One JSON object per line: `id`, `suite`, `split` (`calibration` | `heldout`), `inputs`,
`labels`. Models in `cases.py` forbid unknown fields.

- `retrieval.jsonl`: 25 paraphrased questions with relevant doc ids.
- `answer.jsonl`: 15 answerable questions (expected doc ids, 1-3 key facts, each a list
  of accepted alternatives) and 5 unanswerable ones (`expect_insufficient`).
- `duplicates.jsonl`: 16 labelled duplicates and 7 hard negatives from the dataset
  (calibration: the thresholds were tuned on them), 12 new paraphrases and 10 novel bugs
  (held out; novel bugs carry component and severity for routing).
- `injection.jsonl`: questions retrieving the injected docs, reports making them duplicate
  candidates, a report with its own injected text, and direct user injections.
- Routing is derived: leave-one-out over all 66 dataset bugs plus the 10 novel bugs.

Adding cases: write new wording (no 6-word phrase copied from the corpus; a test checks),
label from the data, never from the scores, and run `uv run pytest -q tests/unit/test_eval_cases.py`.

## Errored vs failed

An errored case (provider or store error) is listed separately and excluded from all
metrics, which are computed over completed cases with the errored count shown. A failure
is a completed case with a wrong result. Re-run until nothing errors before quoting
numbers; cached cases cost nothing.

## Safety

Case text and model output are untrusted data. The judge prompt delimits and neutralises
them; reports are scanned and secret-like strings or configured secret values are
replaced with `[REDACTED]` before writing; the command line in the report has local paths
removed. The harness never imports the GitHub, Slack or proposal modules and never
confirms an action (checked at runtime and by a subprocess test).

## Files

`run.py` CLI · `suites.py` suites · `metrics.py` pure metrics · `judge.py` groundedness and
fact recall · `checks.py` injection assertions and redaction · `llm_cache.py` cache, rate
limiter, token capture · `thresholds.py` / `thresholds.json` floors · `report.py` output ·
`fakes.py` offline doubles · `duplicate_retrieval.py` the preliminary M2 check (kept).
