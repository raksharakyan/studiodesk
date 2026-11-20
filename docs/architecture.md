# Architecture notes

This adds detail to the diagram and the request walkthroughs in the [README](../README.md#architecture). The threat model and controls are in [SECURITY.md](../SECURITY.md).

## Process model

The app is a single FastAPI process. `main.py::create_app` is cheap: the embedding model, the Qdrant client, the LLM client and the outbound `httpx.Client` are created in the lifespan startup, and only if a test did not inject them. They are closed again on shutdown. Every collaborator reaches the routers through FastAPI `Depends` in `api/deps.py`, so tests can override any of them.

Some state lives in process memory by design. Each item below names its module and the shared backend it would need:

| State | Module | Scale-out replacement |
|---|---|---|
| Pending proposals and the daily issue cap | `actions/proposals.py` (`InMemoryProposalStore` behind the `ProposalStore` protocol) | Redis with per-key TTLs and an atomic compare-and-set for `claim` |
| Per-IP rate limits | `api/ratelimit.py` (slowapi, in memory) | slowapi `storage_uri` pointing at Redis |
| Global LLM budget and concurrency gate | `llm/limits.py` (`LLMCallBudget`, `LLMConcurrencyGate`) | Shared counter or token bucket |
| Daily voice-session quota | `voice/quota.py` | Shared counter |

## Data flow

**Ingest** (`scripts/ingest.py` → `ingest/pipeline.py`)
- `data/loader.py` validates the dataset against the Pydantic models in `models/documents.py`.
- `ingest/chunking.py` splits each document type differently:
  - a bug report becomes one chunk;
  - a crash log becomes one chunk (summary, exception and top stack frames);
  - a patch note becomes one summary chunk plus one chunk per fix;
  - a markdown doc becomes one chunk per heading section, with overlapping windows for long sections.
- `embeddings.py` embeds the chunks in batches. `vectorstore.py` upserts them with deterministic point ids derived from `doc_id#chunk_index`, so re-running the ingest overwrites points instead of duplicating them.
- Payload fields `doc_type`, `platform`, `severity`, `component`, `version` and the integer `version_num` are indexed for filtering. `duplicate_of` is stored so duplicates can be traced to their canonical original.

**Qdrant modes**
- With `QDRANT_URL` set, the app uses Qdrant Cloud over HTTPS. Prod requires this.
- Otherwise it uses embedded local mode at `QDRANT_LOCAL_PATH` (`.qdrant_data/`, gitignored).
- Tests always use `:memory:` with a fake embedder.

**LLM adapters** (`llm/`)
- `LLMClient.structured` takes a system prompt, a user prompt and a Pydantic schema, and returns a validated instance.
- `groq_client.py` and `anthropic_client.py` each pin their provider host and retry transient failures inside a deadline (`LLM_REQUEST_DEADLINE_S`). They raise typed errors (`LLMUnavailable`, `LLMRefusal`, `LLMTruncated`, `LLMInvalidOutput`), which `api/errors.py` maps to generic 502 or 503 responses.
- `GatedLLM` wraps whichever adapter is selected, so every route gets the same concurrency and budget limits.

## Voice

The agent is described in [`voice/agent.json`](../voice/agent.json): the prompt, the two client tools, built-in tools, turn settings, the duration cap and client overrides. `scripts/setup_voice_agent.py` uses `voice/provision.py` to create or update the agent and its tools idempotently, and it re-asserts auth and overrides on every run.

At runtime:
1. The page calls `GET /voice/session` (`api/voice.py`). The server reserves a daily slot and asks ElevenLabs for a signed `wss://api.elevenlabs.io/v1/convai/...` URL, valid for 15 minutes.
2. The browser (`static/voice.js`, using the pinned `@elevenlabs/client` 1.27.0) starts the session with that URL and registers `ask_studiodesk` and `check_bug_report` as client tools.
3. When the agent calls a tool, the browser sends the same same-origin `/ask` or `/bugs/check` request as the text UI. It returns a compact result to the agent (no snippets, tokens or action ids) and shows the full result on the page.
4. Filing still needs the on-screen Confirm button. No tool can confirm.

## Showcase

The public deliverable is static.
- `scripts/record_showcase.py` runs `/ask` and `/bugs/check` in-process against the real pipeline and writes `site/data/showcase.json`. The schema is in `showcase.py` and reuses the API's own response models.
- `scripts/build_site.py` validates that file, secret-scans `site/` and writes `_site/`.
- `.github/workflows/pages.yml` publishes `_site/` to GitHub Pages.
- `scripts/deploy_hf_space.py` is the path to a live backend; it needs Hugging Face PRO for Docker Spaces.
