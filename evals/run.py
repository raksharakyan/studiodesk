"""Run the StudioDesk eval suites and write a markdown + JSON report.

    uv run python -m evals.run [--suite all|retrieval|answer|duplicates|routing|injection]
                               [--offline] [--qdrant memory|cloud] [--max-rpm N] [--no-cache]

Default (live): Settings from the environment / `.env`, the real pinned MiniLM embedder, a
fresh in-memory Qdrant with the whole synthetic dataset ingested, the configured LLM
provider as answerer and `EVAL_JUDGE_MODEL` (same provider) as groundedness judge. Writes
`evals/reports/latest.md` and `latest.json`.

`--qdrant cloud` searches the collection at `QDRANT_URL` read-only (no ingest, no writes).
`--offline` uses a fake LLM and a hashing embedder, needs no keys or network, and writes
`evals/reports/offline.md/json` (gitignored) so it can never overwrite a real report.

Secrets are never printed: only model names and counts are reported.
"""

import argparse
import datetime as dt
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient

from evals.cases import CASES_DIR, CaseSet, load_cases
from evals.checks import configured_secrets
from evals.fakes import OFFLINE_EMBEDDER, OFFLINE_MODEL, HashingEmbedder, OfflineLLM
from evals.llm_cache import CACHE_DIR, CachedLLM, CallStats, ResponseCache, Throttle, capture_tokens
from evals.report import write_report
from evals.suites import (
    EvalStack,
    RecordingRetriever,
    SuiteResult,
    routing_cases,
    run_answer,
    run_duplicates,
    run_injection,
    run_retrieval,
    run_routing,
)
from studiodesk.config import Settings
from studiodesk.data.loader import load_dataset
from studiodesk.embeddings import Embedder, SentenceTransformerEmbedder
from studiodesk.ingest.pipeline import ingest
from studiodesk.llm import LLMClient, build_llm
from studiodesk.vectorstore import QdrantStore, build_qdrant_client

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic"
REPORTS_DIR = REPO_ROOT / "evals" / "reports"
SUITES = ("retrieval", "answer", "duplicates", "routing", "injection")
LLM_SUITES = frozenset({"answer", "duplicates", "injection"})
DEFAULT_JUDGE_MODELS = {"groq": "qwen/qwen3.8-27b"}
MEMORY_COLLECTION = "studiodesk_eval"
# Evals run in batch, not behind a request handler, so they can wait longer for rate limits.
EVAL_LLM_OVERRIDES = {
    "llm_max_retries": 3,
    "llm_max_retry_wait_s": 60.0,
    "llm_request_deadline_s": 300.0,
}

logger = logging.getLogger("evals")


class EvalSettings(BaseSettings):
    """Eval-only settings (`EVAL_*` in the environment or `.env`)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    eval_judge_model: str | None = Field(default=None, min_length=1, max_length=128)
    eval_max_rpm: int = Field(default=20, ge=1, le=1_000)


class HarnessError(RuntimeError):
    """A configuration problem that stops the run (message is safe to print)."""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--suite",
        action="append",
        choices=("all", *SUITES),
        help="suite to run (repeatable); default: all",
    )
    parser.add_argument("--offline", action="store_true", help="fake LLM + hashing embedder")
    parser.add_argument("--qdrant", choices=("memory", "cloud"), default="memory")
    parser.add_argument("--max-rpm", type=int, default=None, help="override EVAL_MAX_RPM")
    parser.add_argument(
        "--no-cache", action="store_true", help="ignore cached LLM outputs (still refreshes them)"
    )
    parser.add_argument("--judge-model", default=None, help="override EVAL_JUDGE_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="max cases per suite (smoke runs)")
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)
    if args.max_rpm is not None and args.max_rpm <= 0:
        parser.error("--max-rpm must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offline and args.qdrant == "cloud":
        parser.error("--offline cannot be combined with --qdrant cloud")
    return args


def selected_suites(choices: Sequence[str] | None) -> list[str]:
    """Suites in canonical order; `all` (or nothing) selects every suite."""
    if not choices or "all" in choices:
        return list(SUITES)
    return [suite for suite in SUITES if suite in choices]


def git_sha() -> str:
    """HEAD commit (with `+dirty` for uncommitted changes), or `unknown`."""
    try:
        sha = subprocess.run(  # noqa: S603 (fixed argv, no shell)
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}+dirty" if dirty else sha


def judge_model_for(settings: Settings, override: str | None) -> str:
    """Explicit override, else the provider's default judge, else the answerer model."""
    return (
        override or DEFAULT_JUDGE_MODELS.get(settings.llm_provider) or settings.resolved_llm_model
    )


def _limited[T](items: Sequence[T], limit: int | None) -> list[T]:
    return list(items[:limit] if limit else items)


def build_embedder(offline: bool, settings: Settings) -> tuple[Embedder, str, str]:
    """Embedder plus the model name and revision recorded in the report."""
    if offline:
        return HashingEmbedder(), OFFLINE_EMBEDDER, "n/a"
    return (
        SentenceTransformerEmbedder.from_settings(settings),
        settings.embedding_model,
        settings.embedding_model_revision,
    )


def build_store(
    mode: str, settings: Settings, embedder: Embedder, stack: ExitStack
) -> tuple[QdrantStore, str]:
    """Fresh in-memory store with the dataset ingested, or the cloud collection read-only."""
    if mode == "cloud":
        if not settings.qdrant_url:
            raise HarnessError("--qdrant cloud needs QDRANT_URL (and QDRANT_API_KEY)")
        client = build_qdrant_client(settings)
        stack.callback(client.close)
        store = QdrantStore(client, settings.qdrant_collection)
        points = store.count()
        if points == 0:
            raise HarnessError("the cloud collection is empty; run scripts/ingest.py first")
        return (
            store,
            f"cloud collection `{settings.qdrant_collection}` ({points} points, read-only)",
        )
    client = QdrantClient(location=":memory:")
    stack.callback(client.close)
    store = QdrantStore(client, MEMORY_COLLECTION)
    report = ingest(load_dataset(DATA_DIR), embedder, store, settings.embedding_batch_size)
    return store, f"in-memory, fresh ingest of the synthetic dataset ({report.total_chunks} chunks)"


def build_llms(
    args: argparse.Namespace,
    settings: Settings,
    eval_settings: EvalSettings,
    stats: CallStats,
    stack: ExitStack,
) -> tuple[LLMClient, LLMClient, str, str]:
    """Answerer and judge clients (cached + throttled when live) and their labels."""
    if args.offline:
        fakes = [
            CachedLLM(OfflineLLM(), OFFLINE_MODEL, cache=None, throttle=None, stats=stats)
            for _ in range(2)
        ]
        return fakes[0], fakes[1], OFFLINE_MODEL, OFFLINE_MODEL
    llm_settings = settings.model_copy(update=EVAL_LLM_OVERRIDES)
    judge_model = judge_model_for(settings, args.judge_model or eval_settings.eval_judge_model)
    judge_settings = llm_settings.model_copy(update={"llm_model": judge_model})
    answerer = build_llm(llm_settings)
    judge = build_llm(judge_settings)
    if answerer is None or judge is None:
        raise HarnessError(
            f"no API key for LLM_PROVIDER={settings.llm_provider}; set it in the environment "
            "or .env, or run with --offline"
        )
    stack.callback(answerer.close)
    stack.callback(judge.close)
    cache = ResponseCache(args.cache_dir)
    throttle = Throttle(args.max_rpm or eval_settings.eval_max_rpm)
    common: dict[str, Any] = {
        "cache": cache,
        "throttle": throttle,
        "stats": stats,
        "read_cache": not args.no_cache,
    }
    answer_model = settings.resolved_llm_model
    return (
        CachedLLM(answerer, answer_model, **common),
        CachedLLM(judge, judge_model, **common),
        f"{settings.llm_provider}:{answer_model}",
        f"{settings.llm_provider}:{judge_model}",
    )


def run_suites(
    suites: Sequence[str], cases: CaseSet, stack: EvalStack, limit: int | None
) -> dict[str, SuiteResult]:
    """Run the selected suites in order."""
    runners: dict[str, Callable[[], SuiteResult]] = {
        "retrieval": lambda: run_retrieval(_limited(cases.retrieval, limit), stack),
        "answer": lambda: run_answer(_limited(cases.answer, limit), stack),
        "duplicates": lambda: run_duplicates(_limited(cases.duplicates, limit), stack),
        "routing": lambda: run_routing(
            _limited(routing_cases(stack.dataset, cases.duplicates), limit), stack
        ),
        "injection": lambda: run_injection(_limited(cases.injection, limit), stack),
    }
    results: dict[str, SuiteResult] = {}
    for suite in suites:
        started = time.perf_counter()
        results[suite] = runners[suite]()
        print(f"[{suite}] done in {time.perf_counter() - started:.1f}s", flush=True)
    return results


def preflight(args: argparse.Namespace, settings: Settings, suites: Sequence[str]) -> None:
    """Fail fast on missing configuration, before the embedding model is loaded.

    Raises:
        HarnessError: with a message that names the missing setting (never its value).
    """
    if args.qdrant == "cloud" and not settings.qdrant_url:
        raise HarnessError("--qdrant cloud needs QDRANT_URL (and QDRANT_API_KEY)")
    if not args.offline and LLM_SUITES & set(suites) and settings.llm_api_key is None:
        raise HarnessError(
            f"no API key for LLM_PROVIDER={settings.llm_provider}; set it in the environment "
            "or .env, or run with --offline"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evals; returns 0 on success, 2 on a configuration error."""
    args = parse_args(argv)
    suites = selected_suites(args.suite)
    started = time.perf_counter()
    try:
        cases = load_cases(args.cases_dir)
        # Offline runs are hermetic: no .env, so no keys are even loaded.
        settings = Settings(_env_file=None) if args.offline else Settings()
        eval_settings = EvalSettings(_env_file=None) if args.offline else EvalSettings()
        stats = CallStats()
        preflight(args, settings, suites)
        with ExitStack() as resources, capture_tokens(stats):
            answerer: LLMClient | None = None
            judge: LLMClient | None = None
            answerer_label = judge_label = "not used"
            if LLM_SUITES & set(suites):
                answerer, judge, answerer_label, judge_label = build_llms(
                    args, settings, eval_settings, stats, resources
                )
            embedder, embedding_model, embedding_revision = build_embedder(args.offline, settings)
            store, qdrant_label = build_store(args.qdrant, settings, embedder, resources)
            stack = EvalStack(
                settings=settings,
                store=store,
                retriever=RecordingRetriever(embedder, store),
                dataset=load_dataset(DATA_DIR),
                answer_llm=answerer,
                judge_llm=judge,
                secrets=[] if args.offline else configured_secrets(settings),
            )
            results = run_suites(suites, cases, stack, args.limit)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    command = "uv run python -m evals.run " + " ".join(argv if argv is not None else sys.argv[1:])
    meta = {
        "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(),
        "offline": args.offline,
        "command": command.strip(),
        "suites": suites,
        "answerer": answerer_label,
        "judge": judge_label,
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "qdrant": qdrant_label,
        "agent_settings": {
            "answer_top_k": settings.answer_top_k,
            "dup_search_k": settings.dup_search_k,
            "dup_candidate_threshold": settings.dup_candidate_threshold,
            "dup_auto_threshold": settings.dup_auto_threshold,
            "routing_k": settings.routing_k,
        },
        "case_files": cases.fingerprints,
        "limit_per_suite": args.limit,
    }
    cost = {
        **stats.as_dict(),
        "max_rpm": "n/a (offline)" if args.offline else args.max_rpm or eval_settings.eval_max_rpm,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    stem = "offline" if args.offline else "latest"
    md_path, json_path = write_report(args.output_dir, stem, meta, results, cost)
    for name, result in results.items():
        print(f"{name}: {len(result.failures)} failure(s); metrics: {result.metrics}")
    print(f"cost: {cost}")
    print(f"report: {md_path} and {json_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
