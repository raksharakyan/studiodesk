// StudioDesk Voice page.
//
// Security rules (see docs/security/m3-review.md):
// - All server, model and corpus text is rendered with textContent / createElement only.
// - The confirm token lives only in a module variable: never in the DOM, URL or storage.
// - The voice agent has no confirm tool. Confirm and Cancel run only on real button clicks.
// - Fetches are same-origin with a JSON body; no secrets live in this file.

const ASK_MAX_CHARS = 1000;
const TITLE_MIN_CHARS = 5;
const TITLE_MAX_CHARS = 200;
const DESCRIPTION_MAX_CHARS = 4000;
const PLATFORMS = ["pc", "ps5", "xbox_series", "switch"];
const VERSION_RE = /^\d+\.\d+\.\d+$/;
const TRANSCRIPT_MAX_ITEMS = 200;
// A freshly shown proposal cannot be confirmed for this long, so a preview swapped in by the
// voice agent right before a click is never confirmed by accident.
const CONFIRM_ARM_DELAY_MS = 1200;

// Audio worklets vendored from the same pinned SDK version as the script in voice.html and
// served from this origin (see static/vendor/elevenlabs-client-<version>/README.txt for the
// source and sha256 hashes). Loading them by URL means the CSP script-src does not need blob:
// or data: (the SDK's default worklet loading uses those), and same-origin files need no SRI.
// Keep SDK_VERSION in sync with the <script> tag in voice.html and the vendor directory name.
const SDK_VERSION = "1.27.0";
const WORKLET_BASE = `/static/vendor/elevenlabs-client-${SDK_VERSION}`;
const WORKLET_PATHS = Object.freeze({
  rawAudioProcessor: `${WORKLET_BASE}/rawAudioProcessor.js`,
  audioConcatProcessor: `${WORKLET_BASE}/audioConcatProcessor.js`,
});

/** Names must match voice/agent.json exactly (qa-agent checks parity). */
export const CLIENT_TOOL_NAMES = Object.freeze(["ask_studiodesk", "check_bug_report"]);

const PROPOSAL_NOTE = "The user must press Confirm on screen to file the issue.";

const DOC_TYPE_LABELS = {
  bug_report: "Bug report",
  crash_log: "Crash log",
  patch_note: "Patch notes",
  doc: "Support doc",
};

const VERDICT_LABELS = {
  new: "New report: no duplicate found",
  possible_duplicate: "Possible duplicate",
  duplicate: "Duplicate of an existing report",
};

const SLACK_LABELS = {
  sent: "Slack alert sent.",
  skipped: "Slack alert skipped (not configured).",
  failed: "Slack alert failed; the issue was still filed.",
};

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, retryAfter) {
    super(`HTTP ${status}`);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function api(method, path, body) {
  const init = {
    method,
    headers: { Accept: "application/json" },
    credentials: "same-origin",
    cache: "no-store",
    redirect: "error",
  };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError(0, null);
  }
  if (!response.ok) {
    const retry = Number.parseInt(response.headers.get("Retry-After") ?? "", 10);
    throw new ApiError(response.status, Number.isFinite(retry) && retry > 0 ? retry : null);
  }
  try {
    return await response.json();
  } catch {
    throw new ApiError(-1, null);
  }
}

/**
 * Short, speakable message for a failed request. Built only from the status code, never from
 * the response body, so nothing server- or model-controlled reaches the voice agent here.
 * @param {"session"|"ask"|"bugs"|"confirm"|"cancel"} context
 */
export function friendlyError(err, context) {
  const status = err instanceof ApiError ? err.status : -1;
  const wait = err instanceof ApiError && err.retryAfter ? ` in about ${err.retryAfter} seconds` : " in a moment";
  if (status === 0) return "I couldn't reach the StudioDesk server. Please check the connection and try again.";
  if (status === 429) return `Too many requests right now. Please try again${wait}.`;
  if (context === "session") {
    if (status === 503) return "Voice is not configured on this server. You can still type a question below.";
    return "Voice could not be started. Please try again later.";
  }
  if (context === "confirm" || context === "cancel") {
    if (status === 404) return "That issue proposal was not found. Please check the bug again.";
    if (status === 409) return "That issue proposal was already confirmed or cancelled.";
    if (status === 410) return "That issue proposal expired. Please check the bug again.";
    if (status === 503) return "Issue filing is unavailable right now. Nothing was filed.";
    if (status === 502) return "The issue could not be created. Nothing was filed. You can try again.";
    return "Something went wrong with this action. Nothing was filed.";
  }
  if (status === 422) {
    return context === "bugs"
      ? "Some bug details were invalid. The title needs 5 to 200 characters, the version looks like 1.4.0, and the platform is PC, PS5, Xbox Series or Switch."
      : "The question was invalid. Please keep it under 1000 characters.";
  }
  if (status === 503) return `The assistant is temporarily unavailable. Please try again${wait}.`;
  if (status === 502) return "The assistant could not complete this request. Please try rephrasing.";
  return "Something went wrong. Please try again.";
}

// ---------------------------------------------------------------------------
// Pure helpers (exported for tests)
// ---------------------------------------------------------------------------

function asText(value) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

/** Map spoken platform names to the API enum, or null. */
export function normalisePlatform(value) {
  const v = asText(value).trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (PLATFORMS.includes(v)) return v;
  if (/^(windows|windows_pc|steam|computer|desktop)$/.test(v)) return "pc";
  if (/^(ps_?5|playstation_?5|playstation)$/.test(v)) return "ps5";
  if (/^(xbox|xbox_?series(_?[xs]|_x_?s|_x\|s)?|series_[xs])$/.test(v)) return "xbox_series";
  if (/^(nintendo_?switch|nintendo)$/.test(v)) return "switch";
  return null;
}

/** Pull an x.y.z version out of speech like "version 1.4.0" or "v1.4.0"; null if absent. */
export function extractVersion(value) {
  const match = asText(value).match(/\d+\.\d+\.\d+/);
  return match ? match[0] : null;
}

/** Compact result for the voice agent: no snippets, only ids and titles. */
export function toAgentAskResult(data) {
  return JSON.stringify({
    answer: asText(data?.answer),
    sources: (Array.isArray(data?.sources) ? data.sources : []).map((s) => ({
      doc_id: asText(s?.doc_id),
      title: asText(s?.title),
    })),
    insufficient_context: Boolean(data?.insufficient_context),
  });
}

export function isProposalReady(proposal) {
  return Boolean(
    proposal &&
      typeof proposal.action_id === "string" &&
      proposal.action_id &&
      typeof proposal.confirm_token === "string" &&
      proposal.confirm_token &&
      !proposal.actions_disabled_reason,
  );
}

/** Compact result for the voice agent. Never includes the confirm token or action id. */
export function toAgentBugResult(data) {
  const candidates = Array.isArray(data?.candidates) ? data.candidates : [];
  const matched = candidates.find((c) => c?.doc_id && c.doc_id === data?.matched_report);
  const ready = isProposalReady(data?.proposed_action);
  let note = PROPOSAL_NOTE;
  if (!ready) {
    note = data?.proposed_action
      ? "Issue filing is currently disabled, so no issue can be filed from this check."
      : "No new issue is proposed because this is a duplicate of an existing report.";
  }
  return JSON.stringify({
    verdict: asText(data?.verdict),
    duplicate_of: data?.duplicate_of ? asText(data.duplicate_of) : null,
    matched_title: matched ? asText(matched.title) : null,
    labels: (Array.isArray(data?.routing?.labels) ? data.routing.labels : []).map(asText),
    proposal_ready: ready,
    note,
  });
}

/** Only https links to github.com are shown. Returns the normalised href or null. */
export function safeIssueUrl(value) {
  const raw = asText(value);
  if (!raw.startsWith("https://github.com/")) return null;
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:" || url.hostname !== "github.com" || url.username || url.password) return null;
    return url.href;
  } catch {
    return null;
  }
}

const BUG_INPUT_ERRORS = {
  title: {
    agent: "The bug title must be 5 to 200 characters. Please ask the user for a short title.",
    form: "Please enter a title of 5 to 200 characters.",
  },
  description_missing: {
    agent: "Please ask the user to describe what happened.",
    form: "Please describe what happened.",
  },
  description_long: {
    agent: "The description is too long. Please summarise it in under 4000 characters.",
    form: "Please keep the description under 4000 characters.",
  },
  platform: {
    agent: "Please ask which platform: PC, PlayStation 5, Xbox Series or Nintendo Switch.",
    form: "Please choose a platform.",
  },
  version: {
    agent: "Please ask for the game version, in the form major.minor.patch, for example 1.4.0.",
    form: "Please enter the game version as major.minor.patch, for example 1.4.0.",
  },
};

/**
 * Validate tool or form input for /bugs/check.
 * Returns {body} or {error: {agent, form}, field} with speakable/visible messages.
 */
export function buildBugCheckBody(params) {
  const title = asText(params?.title).trim();
  const description = asText(params?.description).trim();
  const platform = normalisePlatform(params?.platform);
  const version = extractVersion(params?.version);
  const fail = (key, field) => ({ error: BUG_INPUT_ERRORS[key], field });
  if (title.length < TITLE_MIN_CHARS || title.length > TITLE_MAX_CHARS) return fail("title", "bug-title");
  if (!description) return fail("description_missing", "bug-description");
  if (description.length > DESCRIPTION_MAX_CHARS) return fail("description_long", "bug-description");
  if (!platform) return fail("platform", "bug-platform");
  if (!version || !VERSION_RE.test(version)) return fail("version", "bug-version");
  return { body: { title, description, platform, version } };
}

// ---------------------------------------------------------------------------
// DOM helpers
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

function setStatus(id, text, isError = false) {
  const node = $(id);
  node.textContent = asText(text);
  node.classList.toggle("is-error", isError);
}

function fillTags(listId, labels) {
  const list = $(listId);
  list.replaceChildren();
  const items = Array.isArray(labels) ? labels : [];
  if (items.length === 0) {
    list.append(el("li", "", "none"));
    return;
  }
  for (const label of items) list.append(el("li", "", label));
}

function formatScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "n/a";
}

// ---------------------------------------------------------------------------
// Ask: rendering and calls
// ---------------------------------------------------------------------------

let askSeq = 0;

function renderAnswer(question, data) {
  show("answer-empty", false);
  show("answer-content", true);
  setText("answer-question", `Question: ${question}`);
  setText("answer-text", data.answer);

  const notes = [];
  if (data.insufficient_context) {
    notes.push("StudioDesk did not find enough information in its sources to answer confidently.");
  }
  const removed = Number(data.removed_citations) || 0;
  if (removed > 0) {
    notes.push(`${removed} citation${removed === 1 ? " was" : "s were"} removed because ${removed === 1 ? "it" : "they"} did not match a retrieved source.`);
  }
  setText("answer-notes", notes.join(" "));
  show("answer-notes", notes.length > 0);

  const list = $("sources");
  list.replaceChildren();
  const sources = Array.isArray(data.sources) ? data.sources : [];
  show("sources-empty", sources.length === 0);
  for (const source of sources) {
    const item = el("li");
    item.append(el("span", "card-title", source.title));
    const type = DOC_TYPE_LABELS[source.doc_type] ?? asText(source.doc_type);
    item.append(el("span", "card-meta", `${asText(source.doc_id)} · ${type} · relevance ${formatScore(source.score)}`));
    if (source.snippet) item.append(el("p", "card-text", source.snippet));
    list.append(item);
  }
}

/** POST /ask and render. Returns {ok, data} or {ok:false, message}. */
async function runAsk(question) {
  const seq = ++askSeq;
  setStatus("ask-status", "Asking StudioDesk…");
  try {
    const data = await api("POST", "/ask", { question });
    if (seq === askSeq) {
      renderAnswer(question, data);
      setStatus("ask-status", data.insufficient_context ? "Answer ready (limited information)." : "Answer ready.");
    }
    return { ok: true, data };
  } catch (err) {
    const message = friendlyError(err, "ask");
    if (seq === askSeq) setStatus("ask-status", message, true);
    return { ok: false, message };
  }
}

// ---------------------------------------------------------------------------
// Bug check: rendering, proposal and confirm/cancel
// ---------------------------------------------------------------------------

// The only place the confirm token is kept. Replaced when a new proposal is rendered.
let currentProposal = null; // { actionId, token, armed, busy, done }
let bugSeq = 0;

function setActionButtons({ confirm, cancel }) {
  $("confirm-btn").disabled = !confirm;
  $("cancel-btn").disabled = !cancel;
}

function renderCandidates(candidates) {
  const list = $("candidates");
  list.replaceChildren();
  show("candidates-empty", candidates.length === 0);
  for (const c of candidates) {
    const item = el("li");
    item.append(el("span", "card-title", c.title));
    const flag = c.is_duplicate ? "flagged as duplicate" : "not a duplicate";
    item.append(
      el("span", "card-meta", `${asText(c.doc_id)} · similarity ${formatScore(c.score)} · ${flag} · confidence ${formatScore(c.confidence)}`),
    );
    if (c.reason) {
      const reason = el("p", "card-text");
      reason.append(el("span", "ai-label", "AI-generated reason: "));
      reason.append(document.createTextNode(asText(c.reason)));
      item.append(reason);
    }
    list.append(item);
  }
}

function renderProposal(proposal) {
  currentProposal = null;
  setStatus("action-status", "");
  show("action-result", false);
  $("action-result").replaceChildren();

  if (!proposal) {
    show("proposal", false);
    return;
  }
  show("proposal", true);
  setText("proposal-repo", proposal.repo);
  setText("proposal-title", proposal.preview?.title);
  fillTags("proposal-labels", proposal.preview?.labels);
  const expires = proposal.expires_at ? new Date(proposal.expires_at) : null;
  setText("proposal-expires", expires && !Number.isNaN(expires.getTime()) ? expires.toLocaleTimeString() : "n/a");
  // Preformatted plain text, exactly as it will be filed.
  setText("proposal-body", proposal.preview?.body);

  if (!isProposalReady(proposal)) {
    const reason = proposal.actions_disabled_reason
      ? `Issue filing is disabled: ${asText(proposal.actions_disabled_reason)}`
      : "This proposal cannot be confirmed.";
    setText("proposal-disabled", reason);
    show("proposal-disabled", true);
    setActionButtons({ confirm: false, cancel: false });
    return;
  }

  show("proposal-disabled", false);
  const state = { actionId: proposal.action_id, token: proposal.confirm_token, armed: false, busy: false, done: false };
  currentProposal = state;
  setActionButtons({ confirm: false, cancel: true });
  window.setTimeout(() => {
    if (currentProposal === state && !state.busy && !state.done) {
      state.armed = true;
      setActionButtons({ confirm: true, cancel: true });
    }
  }, CONFIRM_ARM_DELAY_MS);
}

function renderBugResult(data) {
  show("bug-result", true);
  setText("bug-verdict", VERDICT_LABELS[data.verdict] ?? asText(data.verdict));

  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const parts = [];
  if (data.matched_report) {
    const matched = candidates.find((c) => c.doc_id === data.matched_report);
    parts.push(`Matched report: ${asText(data.matched_report)}${matched ? ` (${asText(matched.title)})` : ""}.`);
  }
  if (data.duplicate_of && data.duplicate_of !== data.matched_report) {
    parts.push(`Original report: ${asText(data.duplicate_of)}.`);
  }
  setText("bug-match", parts.join(" "));
  show("bug-match", parts.length > 0);

  renderCandidates(candidates);
  setText("routing-component", data.routing?.component ?? "unknown");
  setText("routing-severity", data.routing?.severity ?? "unknown");
  fillTags("routing-labels", data.routing?.labels);
  renderProposal(data.proposed_action);
}

/** POST /bugs/check and render. Returns {ok, data} or {ok:false, message}. */
async function runBugCheck(body) {
  const seq = ++bugSeq;
  setStatus("bug-status", "Checking for duplicates…");
  try {
    const data = await api("POST", "/bugs/check", body);
    if (seq === bugSeq) {
      renderBugResult(data);
      const suffix = isProposalReady(data.proposed_action) ? " Review the issue preview and press Confirm to file it." : "";
      setStatus("bug-status", `Bug check complete: ${VERDICT_LABELS[data.verdict] ?? asText(data.verdict)}.${suffix}`);
    }
    return { ok: true, data };
  } catch (err) {
    const message = friendlyError(err, "bugs");
    if (seq === bugSeq) setStatus("bug-status", message, true);
    return { ok: false, message };
  }
}

function tellAgent(text) {
  // Our own fixed text only; lets the voice agent know what happened on screen.
  try {
    conversation?.sendContextualUpdate?.(text);
  } catch {
    /* best effort */
  }
}

async function onConfirmClick(event) {
  const state = currentProposal;
  // Only a trusted user click can confirm (event.isTrusted is false for script-dispatched clicks).
  if (!event.isTrusted || !state || !state.armed || state.busy || state.done) return;
  state.busy = true;
  setActionButtons({ confirm: false, cancel: false });
  setStatus("action-status", "Filing the issue…");
  try {
    const data = await api("POST", `/actions/${encodeURIComponent(state.actionId)}/confirm`, {
      confirm_token: state.token,
    });
    state.done = true;
    state.token = null;
    if (currentProposal === state) currentProposal = null;
    const result = $("action-result");
    result.replaceChildren();
    const number = Number.isInteger(data.issue_number) ? data.issue_number : null;
    const href = safeIssueUrl(data.issue_url);
    result.append(document.createTextNode("Issue filed: "));
    if (href) {
      const link = el("a", "", number ? `issue #${number}` : "view issue");
      link.href = href;
      link.rel = "noopener noreferrer";
      link.target = "_blank";
      result.append(link, document.createTextNode(" (opens in a new tab). "));
    } else {
      result.append(document.createTextNode(number ? `issue #${number}. ` : "the link could not be shown. "));
    }
    result.append(document.createTextNode(SLACK_LABELS[data.slack] ?? ""));
    show("action-result", true);
    // The focused Confirm button is now disabled; keep keyboard focus on the outcome.
    (result.querySelector("a") ?? $("action-status")).focus();
    setStatus("action-status", number ? `Confirmed. Issue #${number} was filed.` : "Confirmed. The issue was filed.");
    tellAgent(number ? `The user pressed Confirm and GitHub issue #${number} was filed.` : "The user pressed Confirm and the issue was filed.");
  } catch (err) {
    state.busy = false;
    const message = friendlyError(err, "confirm");
    setStatus("action-status", message, true);
    const status = err instanceof ApiError ? err.status : -1;
    if ([404, 409, 410].includes(status)) {
      state.done = true;
      state.token = null;
      if (currentProposal === state) currentProposal = null;
      $("action-status").focus();
    } else if (currentProposal === state) {
      setActionButtons({ confirm: true, cancel: true });
    }
  }
}

async function onCancelClick(event) {
  const state = currentProposal;
  if (!event.isTrusted || !state || state.busy || state.done) return;
  state.busy = true;
  setActionButtons({ confirm: false, cancel: false });
  setStatus("action-status", "Cancelling…");
  try {
    await api("POST", `/actions/${encodeURIComponent(state.actionId)}/cancel`, { confirm_token: state.token });
    state.done = true;
    state.token = null;
    if (currentProposal === state) currentProposal = null;
    setStatus("action-status", "Cancelled. Nothing was filed.");
    $("action-status").focus();
    tellAgent("The user pressed Cancel. No issue was filed.");
  } catch (err) {
    state.busy = false;
    setStatus("action-status", friendlyError(err, "cancel"), true);
    const status = err instanceof ApiError ? err.status : -1;
    if ([404, 409, 410].includes(status)) {
      state.done = true;
      state.token = null;
      if (currentProposal === state) currentProposal = null;
      $("action-status").focus();
    } else if (currentProposal === state) {
      setActionButtons({ confirm: state.armed, cancel: true });
    }
  }
}

// ---------------------------------------------------------------------------
// Client tools exposed to the ElevenLabs agent (no confirm tool, by design)
// ---------------------------------------------------------------------------

const clientTools = {
  async ask_studiodesk(params) {
    const question = asText(params?.question).trim();
    if (!question) return "Please ask the user what they want to know.";
    if (question.length > ASK_MAX_CHARS) return "That question is too long. Please shorten it to under 1000 characters.";
    const result = await runAsk(question);
    return result.ok ? toAgentAskResult(result.data) : `Error: ${result.message}`;
  },

  async check_bug_report(params) {
    const built = buildBugCheckBody(params);
    if (built.error) {
      setStatus("bug-status", built.error.form, true);
      return `Error: ${built.error.agent}`;
    }
    const result = await runBugCheck(built.body);
    return result.ok ? toAgentBugResult(result.data) : `Error: ${result.message}`;
  },
};

// ---------------------------------------------------------------------------
// Voice session
// ---------------------------------------------------------------------------

let conversation = null;
let connecting = false;
let stopRequested = false;

function setVoiceError(message) {
  const node = $("voice-error");
  node.textContent = asText(message);
  node.hidden = !message;
}

function setVoiceStatus(status) {
  const labels = {
    disconnected: "Disconnected",
    connecting: "Connecting…",
    connected: "Connected",
    disconnecting: "Disconnecting…",
  };
  $("voice-status").querySelector('[data-role="status-text"]').textContent = labels[status] ?? asText(status);
  const active = status === "connected" || status === "connecting";
  const button = $("voice-toggle");
  button.setAttribute("aria-pressed", active ? "true" : "false");
  button.querySelector('[data-role="toggle-state"]').textContent = active ? "On" : "Off";
  button.querySelector('[data-role="toggle-label"]').textContent = active ? "Stop voice conversation" : "Start voice conversation";
  if (status !== "connected") setMode("idle");
}

function setMode(mode) {
  const line = $("voice-mode");
  line.dataset.mode = mode;
  const text = { speaking: "Agent speaking", listening: "Agent listening", idle: "Agent idle" };
  line.querySelector('[data-role="mode-text"]').textContent = text[mode] ?? "Agent idle";
}

function appendTranscript(role, message) {
  const text = asText(message).trim();
  if (!text) return;
  show("transcript-empty", false);
  const list = $("transcript");
  const isUser = role === "user";
  const item = el("li", isUser ? "from-user" : "from-agent");
  item.append(el("span", "speaker", isUser ? "You" : "Agent"));
  item.append(document.createTextNode(text));
  list.append(item);
  while (list.children.length > TRANSCRIPT_MAX_ITEMS) list.firstElementChild.remove();
  list.scrollTop = list.scrollHeight;
}

async function ensureMicrophone() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    return "This browser cannot use the microphone here. Voice needs a secure (https or localhost) page.";
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // The SDK opens its own stream; this call only obtains the permission.
    for (const track of stream.getTracks()) track.stop();
    return null;
  } catch (err) {
    const name = err?.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "Microphone access was denied. Allow it in your browser's site settings to use voice, or type your question below.";
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      return "No microphone was found. Connect one, or type your question below.";
    }
    if (name === "NotReadableError") {
      return "The microphone is in use by another app. Close it and try again, or type your question below.";
    }
    return "The microphone could not be started. You can type your question below.";
  }
}

async function startVoice() {
  setVoiceError("");
  const sdk = window.ElevenLabsClient;
  if (!sdk?.Conversation?.startSession) {
    setVoiceError("The voice library did not load. Check your connection or content blockers, then reload the page. Typing still works.");
    return;
  }
  connecting = true;
  stopRequested = false;
  setVoiceStatus("connecting");
  try {
    // Ask the server first, so an unconfigured server never triggers a mic prompt.
    let session;
    try {
      session = await api("GET", "/voice/session");
    } catch (err) {
      setVoiceError(friendlyError(err, "session"));
      setVoiceStatus("disconnected");
      return;
    }
    const signedUrl = asText(session?.signed_url);
    if (!signedUrl.startsWith("wss://")) {
      setVoiceError("Voice could not be started. Please try again later.");
      setVoiceStatus("disconnected");
      return;
    }
    if (stopRequested) {
      setVoiceStatus("disconnected");
      return;
    }

    const micError = await ensureMicrophone();
    if (micError) {
      setVoiceError(micError);
      setVoiceStatus("disconnected");
      return;
    }
    if (stopRequested) {
      setVoiceStatus("disconnected");
      return;
    }

    const started = await sdk.Conversation.startSession({
      signedUrl,
      connectionType: "websocket",
      clientTools,
      workletPaths: { ...WORKLET_PATHS },
      onConnect: () => setVoiceStatus("connected"),
      onDisconnect: (details) => {
        conversation = null;
        setVoiceStatus("disconnected");
        if (details?.reason === "error") setVoiceError("The voice connection was lost. Press the button to start again.");
      },
      onError: (message) => {
        console.warn("Voice error:", message);
        setVoiceError("Voice ran into a problem. If it stops responding, stop and start voice again.");
      },
      onStatusChange: ({ status }) => setVoiceStatus(status),
      onModeChange: ({ mode }) => setMode(mode === "speaking" ? "speaking" : "listening"),
      onMessage: ({ role, source, message }) => appendTranscript(role ?? (source === "user" ? "user" : "agent"), message),
    });
    conversation = started;
    if (stopRequested) await stopVoice();
  } catch (err) {
    console.warn("Voice start failed:", err);
    conversation = null;
    setVoiceError("Voice could not be started. Please try again later.");
    setVoiceStatus("disconnected");
  } finally {
    connecting = false;
  }
}

async function stopVoice() {
  const active = conversation;
  conversation = null;
  if (!active) {
    setVoiceStatus("disconnected");
    return;
  }
  setVoiceStatus("disconnecting");
  try {
    await active.endSession();
  } catch (err) {
    console.warn("Voice stop failed:", err);
  }
  setVoiceStatus("disconnected");
}

function onVoiceToggle() {
  if (connecting) {
    stopRequested = true;
    setVoiceStatus("disconnecting");
    return;
  }
  if (conversation) {
    void stopVoice();
  } else {
    void startVoice();
  }
}

// ---------------------------------------------------------------------------
// Forms and wiring
// ---------------------------------------------------------------------------

function onAskSubmit(event) {
  event.preventDefault();
  const input = $("ask-question");
  const question = input.value.trim();
  if (!question) {
    setStatus("ask-status", "Please enter a question.", true);
    input.focus();
    return;
  }
  if (question.length > ASK_MAX_CHARS) {
    setStatus("ask-status", "Please keep the question under 1000 characters.", true);
    input.focus();
    return;
  }
  void runAsk(question);
}

function onBugSubmit(event) {
  event.preventDefault();
  const built = buildBugCheckBody({
    title: $("bug-title").value,
    description: $("bug-description").value,
    platform: $("bug-platform").value,
    version: $("bug-version").value,
  });
  if (built.error) {
    setStatus("bug-status", built.error.form, true);
    $(built.field).focus();
    return;
  }
  void runBugCheck(built.body);
}

function init() {
  $("voice-toggle").addEventListener("click", onVoiceToggle);
  $("ask-form").addEventListener("submit", onAskSubmit);
  $("bug-form").addEventListener("submit", onBugSubmit);
  $("confirm-btn").addEventListener("click", onConfirmClick);
  $("cancel-btn").addEventListener("click", onCancelClick);
  window.addEventListener("pagehide", () => {
    if (conversation) void conversation.endSession();
  });
}

if (typeof document !== "undefined" && document.getElementById("voice-toggle")) init();
