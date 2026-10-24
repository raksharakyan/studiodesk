// StudioDesk recorded showcase (GitHub Pages). Replays real /ask and /bugs/check responses
// from data/showcase.json; there is no backend, no SDK, no microphone and no other network.
//
// Security rules (the page has a meta CSP: script/style/connect 'self' only):
// - Every recorded value is untrusted data: it is rendered with textContent / text nodes only.
//   No HTML sinks, no inline handlers or styles, no storage, no console.
// - The only fetch is the same-origin, relative "data/showcase.json".
// - The only link built from data is the filed issue example, accepted only as
//   https://github.com/raksharakyan/starfall-outpost-issues/issues/<number>.
// - Data values are never used as element ids or class names; chips carry an index.

export const SHOWCASE_URL = "data/showcase.json";
export const SCHEMA_VERSION = 1;
export const SANDBOX_ISSUES_PREFIX = "https://github.com/raksharakyan/starfall-outpost-issues/issues/";
export const BUG_CHECK_IDS = Object.freeze(["duplicate", "new", "injection"]);

const DOC_TYPE_LABELS = Object.freeze({
  bug_report: "Bug report",
  crash_log: "Crash log",
  patch_note: "Patch notes",
  doc: "Support doc",
});

const VERDICT_LABELS = Object.freeze({
  new: "New report: no duplicate found",
  possible_duplicate: "Possible duplicate",
  duplicate: "Duplicate of an existing report",
});

const PLATFORM_LABELS = Object.freeze({
  pc: "PC",
  ps5: "PS5",
  xbox_series: "Xbox Series",
  switch: "Switch",
});

const PROVIDER_LABELS = Object.freeze({ groq: "Groq", anthropic: "Anthropic" });

// ElevenLabs v3 audio tags such as "[happy]" (same rule as the live page's voice.js): a letter,
// then up to 24 letters, spaces or hyphens. Digits are not allowed, so "[BUG-0001]" stays.
const AUDIO_TAG_RE = /\[[a-z][a-z \-]{0,24}\]\s*/gi;
const ISSUE_PATH_RE = /^\/raksharakyan\/starfall-outpost-issues\/issues\/[1-9]\d{0,8}$/;

// ---------------------------------------------------------------------------
// Pure helpers (no DOM; exported for tests/js/showcase.test.mjs)
// ---------------------------------------------------------------------------

export function asText(value) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

const asList = (value) => (Array.isArray(value) ? value : []);
const isObject = (value) => Boolean(value) && typeof value === "object" && !Array.isArray(value);

/** A score with two decimals, or "n/a". */
export function formatScore(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return "n/a";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "n/a";
}

/** "BUG-0002 · Bug report · PC, PS5 · v1.0.0 · relevance 0.91" (platform/version only if sent). */
export function formatSourceMeta(source) {
  const parts = [asText(source?.doc_id)];
  parts.push(DOC_TYPE_LABELS[source?.doc_type] ?? asText(source?.doc_type));
  const raw = Array.isArray(source?.platform) ? source.platform : source?.platform ? [source.platform] : [];
  const platforms = raw.map((p) => PLATFORM_LABELS[p] ?? asText(p)).filter(Boolean);
  if (platforms.length) parts.push(platforms.join(", "));
  const version = asText(source?.version).trim();
  if (version) parts.push(`v${version.replace(/^v/i, "")}`);
  parts.push(`relevance ${formatScore(source?.score)}`);
  return parts.filter(Boolean).join(" · ");
}

/** The notes under an answer: insufficient context and removed citations. */
export function answerNotes(response) {
  const notes = [];
  if (response?.insufficient_context === true) {
    notes.push("StudioDesk did not find enough information in its sources to answer confidently.");
  }
  const removed = Number.isInteger(response?.removed_citations) ? response.removed_citations : 0;
  if (removed > 0) {
    notes.push(
      `${removed} citation${removed === 1 ? " was" : "s were"} removed because ${removed === 1 ? "it" : "they"} did not match a retrieved source.`,
    );
  }
  return notes.join(" ");
}

export function verdictLabel(verdict) {
  return VERDICT_LABELS[verdict] ?? asText(verdict);
}

export function platformLabel(platform) {
  return PLATFORM_LABELS[platform] ?? asText(platform);
}

/** "Matched report: BUG-0006 (title). Original report: BUG-0002." or "". */
export function matchText(response) {
  const candidates = asList(response?.candidates);
  const parts = [];
  if (response?.matched_report) {
    const matched = candidates.find((c) => c?.doc_id === response.matched_report);
    parts.push(`Matched report: ${asText(response.matched_report)}${matched ? ` (${asText(matched.title)})` : ""}.`);
  }
  if (response?.duplicate_of && response.duplicate_of !== response.matched_report) {
    parts.push(`Original report: ${asText(response.duplicate_of)}.`);
  }
  return parts.join(" ");
}

/** "BUG-0006 · similarity 0.78 · flagged as duplicate · confidence 0.95". */
export function candidateMeta(candidate) {
  const flag = candidate?.is_duplicate === true ? "flagged as duplicate" : "not a duplicate";
  return `${asText(candidate?.doc_id)} · similarity ${formatScore(candidate?.score)} · ${flag} · confidence ${formatScore(candidate?.confidence)}`;
}

/** True when the recorded duplicate check can show the recorded "File it anyway" step. */
export function showsFileAnyway(check, fileAnyway) {
  const verdict = check?.response?.verdict;
  return Boolean(
    check?.id === "duplicate" &&
      (verdict === "duplicate" || verdict === "possible_duplicate") &&
      !check.response.proposed_action &&
      isObject(fileAnyway?.response) &&
      isObject(fileAnyway.response.proposed_action),
  );
}

/** The filed issue example as {href, text} if its URL is a sandbox repo issue, else null. */
export function filedIssueLink(example) {
  const raw = asText(example?.url);
  if (!raw.startsWith(SANDBOX_ISSUES_PREFIX)) return null;
  let url;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  if (url.protocol !== "https:" || url.hostname !== "github.com" || url.port) return null;
  if (url.username || url.password || url.search || url.hash) return null;
  if (!ISSUE_PATH_RE.test(url.pathname) || url.href !== raw) return null;
  const number = url.pathname.split("/").pop();
  const title = asText(example?.title).trim();
  return { href: url.href, text: title ? `${title} (issue #${number})` : `Issue #${number}` };
}

/**
 * "openai/gpt-oss-120b on Groq" from "groq/openai/gpt-oss-120b" (the recorder's
 * provider/model label) or "groq:openai/gpt-oss-120b"; other values are shown as given.
 */
export function llmLabel(value) {
  const raw = asText(value).trim();
  const match = raw.match(/^([a-z]+)[:/](.+)$/);
  if (match && PROVIDER_LABELS[match[1]]) return `${match[2]} on ${PROVIDER_LABELS[match[1]]}`;
  return raw || "its LLM";
}

/** "24 October 2026" (UTC) from an ISO timestamp, or null when it is not a valid date. */
export function recordedDate(iso) {
  const raw = asText(iso);
  if (!/^\d{4}-\d{2}-\d{2}/.test(raw)) return null;
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString("en-GB", { timeZone: "UTC", day: "numeric", month: "long", year: "numeric" });
}

/** The banner's variable parts. */
export function bannerParts(data) {
  return {
    llm: llmLabel(data?.models?.llm),
    date: recordedDate(data?.recorded_at) ?? "the recording date",
  };
}

/**
 * An eval score as shown: ratios in [0, 1] with two decimals, and pass counts such as
 * `injection_pass: "14/14"` as "14 / 14". Anything else is "n/a".
 */
export function formatEval(value) {
  const count = typeof value === "string" ? value.match(/^(\d{1,4})\/(\d{1,4})$/) : null;
  if (count) return `${Number(count[1])} / ${Number(count[2])}`;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) return "n/a";
  return value.toFixed(2);
}

/** Text for each data-eval slot on the page. */
export function evalTexts(evals) {
  const e = isObject(evals) ? evals : {};
  return {
    retrieval_hit_at_5: formatEval(e.retrieval_hit_at_5),
    citation_validity: formatEval(e.citation_validity),
    duplicates_heldout: `${formatEval(e.duplicates_heldout_precision)} / ${formatEval(e.duplicates_heldout_recall)}`,
    injection_pass: formatEval(e.injection_pass),
    routing_component_accuracy: formatEval(e.routing_component_accuracy),
  };
}

/** A count from `dataset`, or null to keep the page default. */
export function datasetCount(dataset, key) {
  const n = dataset?.[key];
  return Number.isInteger(n) && n >= 0 ? String(n) : null;
}

export function stripAudioTags(value) {
  return asText(value).replace(AUDIO_TAG_RE, "").replace(/ {2,}/g, " ").trim();
}

/** Transcript lines as [{isUser, speaker, text}]; unknown roles and empty lines are dropped. */
export function transcriptLines(transcript) {
  const lines = [];
  for (const item of asList(transcript)) {
    if (item?.role !== "user" && item?.role !== "agent") continue;
    const isUser = item.role === "user";
    const text = isUser ? asText(item.text).trim() : stripAudioTags(item.text);
    if (text) lines.push({ isUser, speaker: isUser ? "User" : "Agent", text });
  }
  return lines;
}

/** The recorded question entries that have a usable /ask response. */
export function usableQuestions(data) {
  return asList(data?.questions).filter(
    (q) => isObject(q) && asText(q.question).trim() && isObject(q.response) && typeof q.response.answer === "string",
  );
}

/** The recorded bug check with this id (one of BUG_CHECK_IDS) if it has a usable response. */
export function findBugCheck(data, id) {
  if (!BUG_CHECK_IDS.includes(id)) return null;
  const check = asList(data?.bug_checks).find((c) => isObject(c) && c.id === id);
  return check && isObject(check.request) && isObject(check.response) && check.response.verdict ? check : null;
}

/** True for a question that was declined (no sources, insufficient context). */
export function isDeclined(question) {
  return question?.response?.insufficient_context === true && asList(question.response.sources).length === 0;
}

/** Validate the top level of showcase.json; throws on an unknown schema. */
export function checkShowcase(data) {
  if (!isObject(data) || data.schema !== SCHEMA_VERSION) throw new Error("unsupported showcase data");
  return data;
}

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = asText(text);
  return node;
}

function setText(id, text) {
  $(id).textContent = asText(text);
}

function show(id, visible = true) {
  $(id).hidden = !visible;
}

function fillTags(listId, labels) {
  const list = $(listId);
  list.replaceChildren();
  const items = asList(labels);
  if (items.length === 0) {
    list.append(el("li", "", "none"));
    return;
  }
  for (const label of items) list.append(el("li", "", label));
}

function prefersReducedMotion() {
  return Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches);
}

/** Scroll to a section and move focus to its result heading (for keyboard and screen readers). */
function reveal(section, heading) {
  section?.scrollIntoView?.({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  heading?.focus?.({ preventScroll: true });
}

function setPressed(selector, active) {
  for (const button of document.querySelectorAll(selector)) {
    button.setAttribute("aria-pressed", button === active ? "true" : "false");
  }
}

let showcase = null;

// --- questions -------------------------------------------------------------------

function renderAnswer(entry) {
  const response = entry.response;
  show("answer-empty", false);
  show("answer-content", true);
  setText("answer-question", `Question: ${asText(entry.question)}`);
  setText("answer-text", response.answer);
  const notes = answerNotes(response);
  setText("answer-notes", notes);
  show("answer-notes", Boolean(notes));

  const list = $("sources");
  list.replaceChildren();
  const sources = asList(response.sources);
  show("sources-empty", sources.length === 0);
  for (const source of sources) {
    const item = el("li");
    item.append(el("span", "card-title", source?.title));
    item.append(el("span", "card-meta", formatSourceMeta(source)));
    if (source?.snippet) item.append(el("p", "card-text", source.snippet));
    list.append(item);
  }
}

function onQuestionChip(event) {
  const button = event.currentTarget;
  const entry = usableQuestions(showcase)[Number(button?.dataset?.index)];
  if (!entry) return;
  setPressed("button[data-index]", button);
  renderAnswer(entry);
  setText("ask-status", `Showing the recorded answer to: ${asText(entry.question)}`);
  reveal($("ask"), $("answer-question-heading"));
}

function renderQuestionChips(questions) {
  const list = $("question-chips");
  list.replaceChildren();
  questions.forEach((entry, index) => {
    const declined = isDeclined(entry);
    const button = el("button", declined ? "chip chip-alt" : "chip", entry.question);
    button.type = "button";
    button.dataset.index = String(index);
    button.setAttribute("aria-pressed", "false");
    if (declined) {
      button.append(document.createTextNode(" "));
      button.append(el("span", "chip-note", "(shows it declines questions its sources can't answer)"));
    }
    button.addEventListener("click", onQuestionChip);
    const item = el("li");
    item.append(button);
    list.append(item);
  });
}

// --- bug checks ------------------------------------------------------------------

function renderCandidates(candidates) {
  const list = $("candidates");
  list.replaceChildren();
  show("candidates-empty", candidates.length === 0);
  for (const c of candidates) {
    const item = el("li");
    item.append(el("span", "card-title", c?.title));
    item.append(el("span", "card-meta", candidateMeta(c)));
    if (c?.reason) {
      const reason = el("p", "card-text");
      reason.append(el("span", "ai-label", "AI-generated reason: "));
      reason.append(document.createTextNode(asText(c.reason)));
      item.append(reason);
    }
    list.append(item);
  }
}

function renderProposal(proposal) {
  show("proposal", Boolean(proposal));
  if (!proposal) return;
  setText("proposal-repo", proposal.repo);
  setText("proposal-title", proposal.preview?.title);
  fillTags("proposal-labels", proposal.preview?.labels);
  setText("proposal-body", proposal.preview?.body);
}

function renderBug(request, response, { fileAnywayStep = false, offerFileAnyway = false } = {}) {
  show("bug-empty", false);
  show("bug-result", true);
  setText("report-title", request.title);
  setText("report-description", request.description);
  setText("report-platform", platformLabel(request.platform));
  setText("report-version", request.version);
  show("report-file-anyway", fileAnywayStep);

  setText("bug-verdict", verdictLabel(response.verdict));
  const match = matchText(response);
  setText("bug-match", match);
  show("bug-match", Boolean(match));

  show("file-anyway", offerFileAnyway);
  $("file-anyway-btn").disabled = !offerFileAnyway;

  renderCandidates(asList(response.candidates));
  setText("routing-component", response.routing?.component ?? "unknown");
  setText("routing-severity", response.routing?.severity ?? "unknown");
  fillTags("routing-labels", response.routing?.labels);

  const proposal = isObject(response.proposed_action) ? response.proposed_action : null;
  show("no-proposal", !proposal && !offerFileAnyway);
  renderProposal(proposal);
}

function onBugChip(event) {
  const button = event.currentTarget;
  const check = findBugCheck(showcase, asText(button?.dataset?.bug));
  if (!check) return;
  setPressed("button[data-bug]", button);
  const offer = showsFileAnyway(check, showcase.file_anyway);
  renderBug(check.request, check.response, { offerFileAnyway: offer });
  setText("bug-status", `Showing the recorded check: ${verdictLabel(check.response.verdict)}.`);
  reveal($("bug-panel"), $("bug-report-heading"));
}

function onFileAnyway() {
  const check = findBugCheck(showcase, "duplicate");
  if (!check || !showsFileAnyway(check, showcase.file_anyway)) return;
  const step = showcase.file_anyway;
  const request = isObject(step.request) ? step.request : check.request;
  renderBug(request, step.response, { fileAnywayStep: true });
  setText("bug-status", "Showing the recorded File it anyway result: an issue preview labelled possible-duplicate, which still needs Confirm.");
  reveal($("proposal"), $("proposal-heading"));
}

function renderFiledExample(example) {
  const link = filedIssueLink(example);
  const anchor = $("filed-issue-link");
  if (link) {
    setText("replay-filing-text", "Filing is disabled in the recorded demo — here's a real issue StudioDesk filed:");
    anchor.href = link.href;
    anchor.textContent = link.text;
    anchor.hidden = false;
  } else {
    setText("replay-filing-text", "Filing is disabled in the recorded demo.");
    anchor.hidden = true;
  }
}

// --- static parts ------------------------------------------------------------------

function renderStatic(data) {
  const banner = bannerParts(data);
  setText("banner-llm", banner.llm);
  setText("banner-date", banner.date);

  const counts = { "count-bugs": "bug_reports", "count-crashes": "crash_logs", "count-patches": "patch_notes", "count-docs": "docs" };
  for (const [id, key] of Object.entries(counts)) {
    const value = datasetCount(data.dataset, key);
    if (value !== null) setText(id, value);
  }

  const evals = evalTexts(data.evals);
  for (const node of document.querySelectorAll("[data-eval]")) {
    node.textContent = evals[node.dataset.eval] ?? "n/a";
  }

  const list = $("transcript");
  list.replaceChildren();
  const lines = transcriptLines(data.voice_transcript);
  for (const line of lines) {
    const item = el("li", line.isUser ? "from-user" : "from-agent");
    item.append(el("span", "speaker", line.speaker));
    item.append(document.createTextNode(line.text));
    list.append(item);
  }
  setText("transcript-empty", lines.length ? "" : "No voice session was recorded.");
  show("transcript-empty", lines.length === 0);

  renderFiledExample(data.filed_issue_example);
}

// --- loading -----------------------------------------------------------------------

async function fetchShowcase() {
  const response = await fetch(SHOWCASE_URL, {
    method: "GET",
    headers: { Accept: "application/json" },
    credentials: "same-origin",
    cache: "no-cache",
    redirect: "error",
  });
  if (!response.ok) throw new Error("showcase unavailable");
  return checkShowcase(await response.json());
}

function onLoaded(data) {
  showcase = data;
  renderStatic(data);
  const questions = usableQuestions(data);
  renderQuestionChips(questions);
  for (const button of document.querySelectorAll("button[data-bug]")) {
    const available = Boolean(findBugCheck(data, asText(button.dataset.bug)));
    button.disabled = !available;
    if (available) button.addEventListener("click", onBugChip);
  }
  $("file-anyway-btn").addEventListener("click", onFileAnyway);

  // Show the first examples straight away, without moving focus or scrolling.
  if (questions[0]) {
    renderAnswer(questions[0]);
    setPressed("button[data-index]", document.querySelector('button[data-index="0"]'));
  }
  const first = BUG_CHECK_IDS.map((id) => findBugCheck(data, id)).find(Boolean);
  if (first) {
    renderBug(first.request, first.response, { offerFileAnyway: showsFileAnyway(first, data.file_anyway) });
    setPressed("button[data-bug]", document.querySelector(`button[data-bug="${first.id}"]`));
  }
  setText("load-status", "Recorded results loaded. Pick an example to see what StudioDesk returned.");
}

function onLoadFailed() {
  setText("load-status", "");
  show("load-error", true);
}

function init() {
  fetchShowcase().then(onLoaded).catch(onLoadFailed);
}

if (typeof document !== "undefined" && document.getElementById("question-chips")) init();
