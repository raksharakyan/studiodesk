"""Render the eval results as `latest.md` (for people) and `latest.json` (for tools).

Model and corpus text that reaches the report (failure reasons, unsupported claims) is
escaped for markdown tables and truncated; it is data, never markup we rely on.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.suites import SuiteResult

MAX_CELL_CHARS = 220

HOW_TO_READ = """\
## How to read this

- **Splits.** `calibration` cases are the dataset's own labelled duplicates and hard
  negatives: the duplicate thresholds (0.48 candidate, 0.70 auto) were tuned on them, so
  their scores are optimistic. `heldout` cases were written fresh for M4 and never used for
  tuning; they are the honest estimate of generalisation. Retrieval and answer questions
  are all held out (new wording, not copied from the documents).
- **Retrieval.** Top 5 chunks per question with no filters; doc ids are deduplicated, so a
  multi-chunk doc counts once at its best rank. hit@5 = any relevant doc retrieved;
  MRR@5 = mean of 1/rank of the first relevant doc; recall@5 = share of *all* labelled
  relevant docs retrieved. Questions often have 4-7 relevant docs, so recall@5 cannot reach
  1.0 for them; read it comparatively, not as an absolute.
- **Answer.** Groundedness is judged by a different model family than the answerer:
  the judge splits each answer into claims and marks each as supported by the *cited*
  sources (a claim counts only if it names a source the judge was shown). Fact recall is a
  deterministic, case-insensitive whole-token match of 1-3 key facts per question (each
  with listed alternatives), no judge involved. Citation validity = every id in the answer
  text and sources was retrieved (enforced server-side; `removed_citations` counts ids the
  model invented and the server stripped). Correct refusal = `insufficient_context` on the
  5 unanswerable questions.
- **Duplicates.** "Positive" = `duplicate` or `possible_duplicate` (both show the
  candidate to the user). Precision/recall/F1 are for that binary call; *strict precision*
  is the share of `duplicate` verdicts (which withhold the issue proposal) that point at
  the right original; *canonical accuracy* is the share of true positives whose original
  is correct; *candidate recall* is how often the original made the shortlist at all
  (score >= 0.48), which bounds recall.
- **Routing.** kNN vote over the 7 most similar bug reports, no LLM. `dataset_loo` routes
  each of the 66 dataset bugs with itself excluded; `novel` routes the 10 hand-labelled new
  bugs. Macro-F1 averages over the classes present in labels or predictions.
- **Injection.** Each case must pass every assertion: the injected doc was actually
  retrieved (or shortlisted), no secret-like patterns or configured secret values, no
  system-prompt phrases, only retrieved ids in answers and sources, no forced `duplicate`
  verdict (only allowed when the score gate is met *and* the report genuinely describes
  the injected bug), and no side effects (the store is unchanged; the harness has no
  GitHub/Slack client and never confirms an action).
- **LLM non-determinism.** Answers come from a sampled model; small score changes between
  runs are noise. Cached outputs make re-runs identical until a prompt or model changes.
"""


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 3] + "..."


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "|" + "---|" * len(headers),
    ]
    lines += ["| " + " | ".join(_cell(_fmt(v)) for v in row) + " |" for row in rows]
    return lines


def summary_rows(results: dict[str, SuiteResult]) -> list[list[object]]:
    """One row per suite: name, cases, headline metrics, failure count."""
    rows: list[list[object]] = []
    for name, result in results.items():
        m = result.metrics
        if name == "retrieval":
            headline = (
                f"hit@5 {_fmt(m['hit@5'])}, MRR@5 {_fmt(m['mrr@5'])}, "
                f"recall@5 {_fmt(m['recall@5'])}"
            )
            cases = m["cases"]
        elif name == "answer":
            headline = (
                f"groundedness {_fmt(m['groundedness_mean'])}, "
                f"fact recall {_fmt(m['fact_recall_mean'])}, "
                f"citation validity {_fmt(m['citation_validity'])}, "
                f"correct refusal {_fmt(m['correct_refusal_rate'])}"
            )
            cases = m["cases"]
        elif name == "duplicates":
            cal, held = m["calibration"], m["heldout"]
            headline = (
                f"F1 calibration {_fmt(cal['f1'])} / heldout {_fmt(held['f1'])}; "
                f"strict precision {_fmt(m['overall']['strict_precision_duplicate'])}"
            )
            cases = m["overall"]["cases"]
        elif name == "routing":
            loo, novel = m["dataset_loo"], m["novel"]
            headline = (
                f"component acc LOO {_fmt(loo.get('component_accuracy'))} / "
                f"novel {_fmt(novel.get('component_accuracy'))}; "
                f"severity acc LOO {_fmt(loo.get('severity_accuracy'))}"
            )
            cases = m["all"]["cases"]
        else:
            headline = f"pass rate {_fmt(m['pass_rate'])} ({m['passed']}/{m['cases']})"
            cases = m["cases"]
        rows.append([name, cases, headline, len(result.failures)])
    return rows


def _retrieval_section(result: SuiteResult) -> list[str]:
    m = result.metrics
    lines = ["## Retrieval", ""]
    lines += _table(
        ["Cases", "hit@5", "MRR@5", "recall@5", "hit@1"],
        [[m["cases"], m["hit@5"], m["mrr@5"], m["recall@5"], m["hit@1"]]],
    )
    lines += ["", "<details><summary>Per case</summary>", ""]
    lines += _table(
        ["Case", "Hit", "RR", "Recall", "Retrieved (top 5 chunks, deduplicated)"],
        [
            [r["id"], int(r["hit"]), r["rr"], r["recall"], ", ".join(r["retrieved"])]
            for r in result.cases
        ],
    )
    return [*lines, "", "</details>", ""]


def _answer_section(result: SuiteResult) -> list[str]:
    m = result.metrics
    lines = ["## Answer quality", ""]
    lines += _table(["Metric", "Value"], [[k, v] for k, v in m.items()])
    lines += ["", "<details><summary>Per case</summary>", ""]
    lines += _table(
        ["Case", "Answerable", "Insufficient", "Cited", "Groundedness", "Fact recall", "Removed"],
        [
            [
                r["id"],
                r["answerable"],
                r.get("insufficient"),
                ", ".join(r.get("cited", [])) or r.get("error", ""),
                r.get("groundedness"),
                r.get("fact_recall"),
                r.get("removed_citations"),
            ]
            for r in result.cases
        ],
    )
    return [*lines, "", "</details>", ""]


def _duplicates_section(result: SuiteResult) -> list[str]:
    m = result.metrics
    keys = [k for k in m["overall"] if k in m["calibration"]]
    lines = ["## Duplicate detection", "", "Calibration vs held-out:", ""]
    lines += _table(
        ["Metric", "Calibration", "Held-out", "Overall"],
        [[k, m["calibration"][k], m["heldout"][k], m["overall"][k]] for k in keys],
    )
    table = result.tables["verdicts_by_kind"]
    lines += ["", "Verdicts by case kind:", ""]
    lines += _table(
        ["Kind", *table["columns"]],
        [[kind, *row] for kind, row in zip(table["rows"], table["matrix"], strict=True)],
    )
    lines += ["", "<details><summary>Per case</summary>", ""]
    lines += _table(
        ["Case", "Split", "Expected", "Verdict", "Predicted original", "Original score"],
        [
            [
                r["id"],
                r["split"],
                f"{r['expected']} {r['original'] or ''}".strip(),
                r["verdict"],
                r.get("predicted_original"),
                r.get("original_score"),
            ]
            for r in result.cases
        ],
    )
    return [*lines, "", "</details>", ""]


def _matrix(title: str, confusion: dict[str, Any]) -> list[str]:
    labels = confusion["labels"]
    lines = ["", f"{title} (rows = true, columns = predicted):", ""]
    lines += _table(
        ["true \\ pred", *labels],
        [[label, *row] for label, row in zip(labels, confusion["matrix"], strict=True)],
    )
    return lines


def _routing_section(result: SuiteResult) -> list[str]:
    m = result.metrics
    keys = [
        "cases",
        "component_accuracy",
        "component_macro_f1",
        "severity_accuracy",
        "severity_macro_f1",
    ]
    lines = ["## Routing", ""]
    lines += _table(
        ["Metric", "Dataset LOO", "Novel", "All"],
        [[k, m["dataset_loo"].get(k), m["novel"].get(k), m["all"].get(k)] for k in keys],
    )
    lines += _matrix("Component confusion matrix, all cases", result.tables["component_confusion"])
    lines += _matrix("Severity confusion matrix, all cases", result.tables["severity_confusion"])
    return [*lines, ""]


def _injection_section(result: SuiteResult) -> list[str]:
    m = result.metrics
    lines = ["## Prompt injection", ""]
    lines += _table(
        ["Cases", "Passed", "Pass rate", "Actions executed", "`duplicate` on injected report"],
        [
            [
                m["cases"],
                m["passed"],
                m["pass_rate"],
                m["actions_executed"],
                m["duplicate_verdicts_on_injected"],
            ]
        ],
    )
    lines += ["", "Per assertion:", ""]
    lines += _table(["Assertion", "Pass rate"], list(m["per_assertion_pass_rate"].items()))
    lines += ["", "Per case:", ""]
    lines += _table(
        ["Case", "Mode", "Outcome", "All passed"],
        [
            [
                r["id"],
                r.get("mode", ""),
                _injection_outcome(r),
                all(r["assertions"].values()),
            ]
            for r in result.cases
        ],
    )
    return [*lines, ""]


def _injection_outcome(row: dict[str, Any]) -> str:
    if "error" in row:
        return str(row["error"])
    if row.get("mode") == "bug_check":
        score = row.get("matched_score")
        return (
            f"verdict {row['verdict']}, matched {row.get('matched_report') or '-'}"
            f" (score {_fmt(score)}, gate met {row.get('gate_met')})"
        )
    return (
        f"insufficient={row.get('insufficient')}, cited {', '.join(row.get('cited', [])) or '-'},"
        f" removed {row.get('removed_citations')}"
    )


SECTIONS = {
    "retrieval": _retrieval_section,
    "answer": _answer_section,
    "duplicates": _duplicates_section,
    "routing": _routing_section,
    "injection": _injection_section,
}


def render_markdown(
    meta: dict[str, Any], results: dict[str, SuiteResult], cost: dict[str, Any]
) -> str:
    """The full markdown report."""
    lines = ["# StudioDesk eval report", ""]
    if meta["offline"]:
        lines += [
            "> **OFFLINE RUN: scores are meaningless.** A fake LLM and a hashing embedder were",
            "> used to exercise the harness without keys or network. Do not quote these numbers.",
            "",
        ]
    lines += [
        f"- Date (UTC): {meta['date']}",
        f"- Git SHA: `{meta['git_sha']}`",
        f"- Command: `{meta['command']}`",
        f"- Answerer: `{meta['answerer']}`",
        f"- Judge: `{meta['judge']}`",
        f"- Embedding: `{meta['embedding_model']}` @ `{meta['embedding_revision']}`",
        f"- Qdrant: {meta['qdrant']}",
        f"- Agent settings: {', '.join(f'{k}={v}' for k, v in meta['agent_settings'].items())}",
        f"- Case files: {', '.join(f'{k} `{v}`' for k, v in meta['case_files'].items())}",
        "",
        "## Summary",
        "",
    ]
    lines += _table(["Suite", "Cases", "Headline", "Failures"], summary_rows(results))
    lines.append("")
    for name, result in results.items():
        lines += SECTIONS[name](result)
    lines += ["## Failures", ""]
    any_failures = False
    for name, result in results.items():
        if not result.failures:
            continue
        any_failures = True
        lines += [f"### {name} ({len(result.failures)})", ""]
        lines += [f"- `{case_id}`: {_cell(reason)}" for case_id, reason in result.failures]
        lines.append("")
    if not any_failures:
        lines += ["- none", ""]
    lines += ["## Run cost", ""]
    lines += _table(["Item", "Value"], [[k, v] for k, v in cost.items()])
    lines += ["", HOW_TO_READ]
    return "\n".join(lines)


def to_json(meta: dict[str, Any], results: dict[str, SuiteResult], cost: dict[str, Any]) -> str:
    """Machine-readable report with every per-case row."""
    payload = {
        "meta": meta,
        "summary": [
            {"suite": row[0], "cases": row[1], "headline": row[2], "failures": row[3]}
            for row in summary_rows(results)
        ],
        "suites": {
            name: {
                "metrics": r.metrics,
                "tables": r.tables,
                "failures": [{"id": i, "reason": reason} for i, reason in r.failures],
                "cases": r.cases,
            }
            for name, r in results.items()
        },
        "cost": cost,
    }
    return json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"


def write_report(
    out_dir: Path,
    stem: str,
    meta: dict[str, Any],
    results: dict[str, SuiteResult],
    cost: dict[str, Any],
) -> tuple[Path, Path]:
    """Write `<stem>.md` and `<stem>.json` into `out_dir`; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md_path.write_text(render_markdown(meta, results, cost), encoding="utf-8")
    json_path.write_text(to_json(meta, results, cost), encoding="utf-8")
    return md_path, json_path
