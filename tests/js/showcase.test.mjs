// Tests for the recorded GitHub Pages showcase script site/showcase.js.
// Usage: node tests/js/showcase.test.mjs <path-to-showcase.js> [<path-to-showcase.json>]
// Prints one JSON object per test on stdout and exits 1 if any test failed. No network:
// the DOM test replaces fetch with an in-memory reply and records every URL requested.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { FakeElement, makeRunner } from "./dom_shim.mjs";

const [scriptPath, dataPath] = process.argv.slice(2);
if (!scriptPath) {
  console.error("usage: showcase.test.mjs <showcase.js> [<showcase.json>]");
  process.exit(2);
}

// 1. The real file imports cleanly in node with no browser globals (init is guarded).
const mod = await import(pathToFileURL(scriptPath).href);
const { test, run } = makeRunner();

const SANDBOX = "https://github.com/raksharakyan/starfall-outpost-issues/issues/";

test("exports the documented helpers", () => {
  for (const name of [
    "asText", "formatScore", "formatSourceMeta", "answerNotes", "verdictLabel", "platformLabel",
    "matchText", "candidateMeta", "showsFileAnyway", "filedIssueLink", "llmLabel", "recordedDate",
    "bannerParts", "formatEval", "evalTexts", "datasetCount", "stripAudioTags", "transcriptLines",
    "usableQuestions", "findBugCheck", "isDeclined", "checkShowcase",
  ]) {
    assert.equal(typeof mod[name], "function", name);
  }
  assert.equal(mod.SHOWCASE_URL, "data/showcase.json");
  assert.equal(mod.SANDBOX_ISSUES_PREFIX, SANDBOX);
  assert.deepEqual([...mod.BUG_CHECK_IDS], ["duplicate", "new", "injection"]);
  assert.ok(Object.isFrozen(mod.BUG_CHECK_IDS));
});

test("the showcase URL is relative and same-origin", () => {
  const base = "https://raksharakyan.github.io/studiodesk/";
  const url = new URL(mod.SHOWCASE_URL, base);
  assert.equal(url.origin, "https://raksharakyan.github.io");
  assert.equal(url.pathname, "/studiodesk/data/showcase.json");
});

test("formatScore renders two decimals or n/a", () => {
  assert.equal(mod.formatScore(0.74123), "0.74");
  assert.equal(mod.formatScore(1), "1.00");
  assert.equal(mod.formatScore("0.5"), "0.50");
  for (const bad of [null, undefined, "", "abc", NaN, Infinity, true, {}]) assert.equal(mod.formatScore(bad), "n/a", String(bad));
});

test("formatSourceMeta shows id, type, platforms, version and relevance", () => {
  assert.equal(
    mod.formatSourceMeta({ doc_id: "BUG-0001", doc_type: "bug_report", platform: ["ps5", "pc"], version: "1.0.0", score: 0.7412 }),
    "BUG-0001 · Bug report · PS5, PC · v1.0.0 · relevance 0.74",
  );
  assert.equal(
    mod.formatSourceMeta({ doc_id: "PATCH-1.0.1", doc_type: "patch_note", platform: [], version: "1.0.1", score: 0.6 }),
    "PATCH-1.0.1 · Patch notes · v1.0.1 · relevance 0.60",
  );
  assert.equal(
    mod.formatSourceMeta({ doc_id: "DOC-faq", doc_type: "doc", platform: ["switch"], version: null, score: 0.5 }),
    "DOC-faq · Support doc · Switch · relevance 0.50",
  );
  assert.equal(mod.formatSourceMeta({ doc_id: "X", doc_type: "weird", platform: "toaster", score: "?" }), "X · weird · toaster · relevance n/a");
  assert.equal(mod.formatSourceMeta(null), "relevance n/a");
});

test("answerNotes covers insufficient context and removed citations", () => {
  assert.equal(mod.answerNotes({ insufficient_context: false, removed_citations: 0 }), "");
  assert.equal(mod.answerNotes({ removed_citations: 1 }), "1 citation was removed because it did not match a retrieved source.");
  assert.equal(mod.answerNotes({ removed_citations: 3 }), "3 citations were removed because they did not match a retrieved source.");
  assert.match(mod.answerNotes({ insufficient_context: true, removed_citations: 0 }), /^StudioDesk did not find enough information/);
  assert.equal(mod.answerNotes({ insufficient_context: "yes", removed_citations: "2" }), "");
  assert.equal(mod.answerNotes(undefined), "");
});

test("verdict, match and candidate text", () => {
  assert.equal(mod.verdictLabel("duplicate"), "Duplicate of an existing report");
  assert.equal(mod.verdictLabel("possible_duplicate"), "Possible duplicate");
  assert.equal(mod.verdictLabel("new"), "New report: no duplicate found");
  assert.equal(mod.verdictLabel("<b>x</b>"), "<b>x</b>");
  const response = {
    matched_report: "BUG-0006",
    duplicate_of: "BUG-0002",
    candidates: [{ doc_id: "BUG-0006", title: "Everyone suffocates" }],
  };
  assert.equal(mod.matchText(response), "Matched report: BUG-0006 (Everyone suffocates). Original report: BUG-0002.");
  assert.equal(mod.matchText({ matched_report: "BUG-1", duplicate_of: "BUG-1", candidates: [] }), "Matched report: BUG-1.");
  assert.equal(mod.matchText({ verdict: "new", matched_report: null, duplicate_of: null }), "");
  assert.equal(
    mod.candidateMeta({ doc_id: "BUG-0006", score: 0.7841, is_duplicate: true, confidence: 0.95 }),
    "BUG-0006 · similarity 0.78 · flagged as duplicate · confidence 0.95",
  );
  assert.equal(mod.candidateMeta({ doc_id: "BUG-1", score: 0.5, is_duplicate: "true", confidence: null }), "BUG-1 · similarity 0.50 · not a duplicate · confidence n/a");
});

test("filedIssueLink accepts only sandbox repo issue URLs", () => {
  assert.deepEqual(mod.filedIssueLink({ url: `${SANDBOX}12`, title: "Photo mode camera" }), {
    href: `${SANDBOX}12`,
    text: "Photo mode camera (issue #12)",
  });
  assert.deepEqual(mod.filedIssueLink({ url: `${SANDBOX}7`, title: "  " }), { href: `${SANDBOX}7`, text: "Issue #7" });
  for (const url of [
    "", null, 42, "javascript:alert(1)", "data:text/html,x",
    "http://github.com/raksharakyan/starfall-outpost-issues/issues/1",
    "https://github.com/raksharakyan/studiodesk/issues/1",
    "https://github.com/evil/starfall-outpost-issues/issues/1",
    "https://github.com.evil.example/raksharakyan/starfall-outpost-issues/issues/1",
    "https://user@github.com/raksharakyan/starfall-outpost-issues/issues/1",
    `${SANDBOX}`, `${SANDBOX}0`, `${SANDBOX}abc`, `${SANDBOX}1/../../../evil`, `${SANDBOX}1/comments`,
    `${SANDBOX}1?x=1`, `${SANDBOX}1#frag`, `${SANDBOX}1%2F..`, ` ${SANDBOX}1`, `${SANDBOX}1 `,
    `${SANDBOX}12345678901`,
  ]) {
    assert.equal(mod.filedIssueLink({ url, title: "t" }), null, String(url));
  }
  assert.equal(mod.filedIssueLink(null), null);
});

test("banner parts: LLM label and UTC recording date", () => {
  assert.equal(mod.llmLabel("groq/openai/gpt-oss-120b"), "openai/gpt-oss-120b on Groq");
  assert.equal(mod.llmLabel("groq:openai/gpt-oss-120b"), "openai/gpt-oss-120b on Groq");
  assert.equal(mod.llmLabel("anthropic:claude-sonnet-4-5"), "claude-sonnet-4-5 on Anthropic");
  assert.equal(mod.llmLabel("openai/gpt-oss-120b"), "openai/gpt-oss-120b");
  assert.equal(mod.llmLabel(""), "its LLM");
  assert.equal(mod.recordedDate("2026-10-24T10:30:00Z"), "24 October 2026");
  assert.equal(mod.recordedDate("2026-10-24T23:30:00-05:00"), "25 October 2026");
  for (const bad of ["", "yesterday", "2026-13-45T00:00:00Z", null, 1729760000]) assert.equal(mod.recordedDate(bad), null, String(bad));
  assert.deepEqual(mod.bannerParts({ models: { llm: "groq:m" }, recorded_at: "2026-01-02T00:00:00Z" }), { llm: "m on Groq", date: "2 January 2026" });
  assert.deepEqual(mod.bannerParts({}), { llm: "its LLM", date: "the recording date" });
});

test("eval scores: ratios, injection counts and n/a", () => {
  assert.equal(mod.formatEval(1), "1.00");
  assert.equal(mod.formatEval(0.566), "0.57");
  assert.equal(mod.formatEval("14/14"), "14 / 14");
  assert.equal(mod.formatEval("013/14"), "13 / 14");
  for (const bad of ["1.0", "14 / 14", "14/14 ", "a/b", "12345/1", 1.5, -0.1, NaN, null, { passed: 14, total: 14 }, []]) {
    assert.equal(mod.formatEval(bad), "n/a", JSON.stringify(bad));
  }
  assert.deepEqual(
    mod.evalTexts({
      retrieval_hit_at_5: 1, citation_validity: 1, duplicates_heldout_precision: 1, duplicates_heldout_recall: 0.9,
      injection_pass: "14/14", routing_component_accuracy: 0.566,
    }),
    { retrieval_hit_at_5: "1.00", citation_validity: "1.00", duplicates_heldout: "1.00 / 0.90", injection_pass: "14 / 14", routing_component_accuracy: "0.57" },
  );
  assert.equal(mod.evalTexts(null).duplicates_heldout, "n/a / n/a");
});

test("datasetCount keeps only non-negative integers", () => {
  assert.equal(mod.datasetCount({ docs: 4 }, "docs"), "4");
  assert.equal(mod.datasetCount({ docs: 0 }, "docs"), "0");
  for (const bad of [-1, 1.5, "4", null]) assert.equal(mod.datasetCount({ docs: bad }, "docs"), null, String(bad));
  assert.equal(mod.datasetCount(undefined, "docs"), null);
});

test("transcriptLines strips agent audio tags, keeps ids and drops unknown roles", () => {
  assert.deepEqual(
    mod.transcriptLines([
      { role: "agent", text: "[warm] Hi there.  See [BUG-0001]." },
      { role: "user", text: "  [laughs] really?  " },
      { role: "system", text: "hidden" },
      { role: "agent", text: "[sighs]" },
      { role: "user", text: "" },
      null,
    ]),
    [
      { isUser: false, speaker: "Agent", text: "Hi there. See [BUG-0001]." },
      { isUser: true, speaker: "User", text: "[laughs] really?" },
    ],
  );
  assert.deepEqual(mod.transcriptLines("nope"), []);
});

test("usableQuestions, findBugCheck and isDeclined validate entries", () => {
  const data = {
    questions: [
      { id: "a", question: "Q1", response: { answer: "A", sources: [] } },
      { id: "b", question: "  ", response: { answer: "A" } },
      { id: "c", question: "Q3", response: { answer: 5 } },
      { id: "d", question: "Q4" },
      "junk",
    ],
    bug_checks: [
      { id: "new", request: { title: "t" }, response: { verdict: "new" } },
      { id: "injection", request: { title: "t" }, response: {} },
      { id: "__proto__", request: {}, response: { verdict: "new" } },
    ],
  };
  assert.deepEqual(mod.usableQuestions(data).map((q) => q.id), ["a"]);
  assert.equal(mod.findBugCheck(data, "new").id, "new");
  assert.equal(mod.findBugCheck(data, "injection"), null);
  assert.equal(mod.findBugCheck(data, "duplicate"), null);
  assert.equal(mod.findBugCheck(data, "__proto__"), null);
  assert.equal(mod.isDeclined({ response: { insufficient_context: true, sources: [] } }), true);
  assert.equal(mod.isDeclined({ response: { insufficient_context: true, sources: [{}] } }), false);
  assert.equal(mod.isDeclined({ response: { insufficient_context: false, sources: [] } }), false);
});

test("showsFileAnyway needs the duplicate check without a proposal and a recorded proposal", () => {
  const check = { id: "duplicate", response: { verdict: "duplicate", proposed_action: null } };
  const step = { request: {}, response: { verdict: "duplicate", proposed_action: { repo: "r", preview: {} } } };
  assert.equal(mod.showsFileAnyway(check, step), true);
  assert.equal(mod.showsFileAnyway({ ...check, response: { ...check.response, verdict: "possible_duplicate" } }, step), true);
  assert.equal(mod.showsFileAnyway({ ...check, id: "new" }, step), false);
  assert.equal(mod.showsFileAnyway({ ...check, response: { verdict: "new", proposed_action: null } }, step), false);
  assert.equal(mod.showsFileAnyway({ ...check, response: { verdict: "duplicate", proposed_action: {} } }, step), false);
  assert.equal(mod.showsFileAnyway(check, { response: { proposed_action: null } }), false);
  assert.equal(mod.showsFileAnyway(check, undefined), false);
});

test("checkShowcase accepts schema 1 only", () => {
  const data = { schema: 1 };
  assert.equal(mod.checkShowcase(data), data);
  for (const bad of [null, [], "x", { schema: 2 }, { schema: "1" }, {}]) assert.throws(() => mod.checkShowcase(bad), JSON.stringify(bad));
});

if (dataPath) {
  test("the committed showcase.json renders every example", () => {
    const data = mod.checkShowcase(JSON.parse(readFileSync(dataPath, "utf8")));
    assert.ok(mod.usableQuestions(data).length >= 4, "at least four usable questions");
    assert.equal(mod.usableQuestions(data).length, data.questions.length, "every question is usable");
    for (const id of mod.BUG_CHECK_IDS) assert.ok(mod.findBugCheck(data, id), `bug check ${id}`);
    assert.equal(mod.showsFileAnyway(mod.findBugCheck(data, "duplicate"), data.file_anyway), true);
    assert.ok(mod.filedIssueLink(data.filed_issue_example), "filed issue example is a sandbox issue URL");
    assert.ok(mod.recordedDate(data.recorded_at), "recorded_at is a date");
    assert.ok(mod.transcriptLines(data.voice_transcript).length >= 2, "transcript has lines");
    for (const [key, text] of Object.entries(mod.evalTexts(data.evals))) assert.doesNotMatch(text, /n\/a/, key);
    for (const check of data.bug_checks) {
      const action = check.response.proposed_action;
      if (action) for (const field of ["action_id", "confirm_token", "expires_at"]) assert.equal(action[field], null, `${check.id}.${field}`);
    }
  });
}

// --- DOM: hostile recorded data is rendered as inert text --------------------------------

const HOSTILE = '<img src=x onerror="alert(1)"><script>alert(2)</script>';

function hostileShowcase() {
  const source = { doc_id: "BUG-0001", doc_type: "bug_report", title: HOSTILE, snippet: HOSTILE, score: 0.9, platform: ["pc"], version: "1.0.0" };
  const candidate = { doc_id: "BUG-0002", title: HOSTILE, score: 0.8, is_duplicate: true, confidence: 0.9, reason: HOSTILE };
  const routing = { component: HOSTILE, severity: "high", labels: [HOSTILE] };
  const proposal = { action_id: null, confirm_token: null, expires_at: null, repo: HOSTILE, preview: { title: HOSTILE, body: HOSTILE, labels: [HOSTILE] } };
  const request = { title: HOSTILE, description: HOSTILE, platform: HOSTILE, version: "1.0.0" };
  return {
    schema: 1,
    recorded_at: "2026-10-24T10:30:00Z",
    models: { llm: HOSTILE, embedding: "e" },
    dataset: { bug_reports: HOSTILE, crash_logs: 25, patch_notes: 10, docs: 4 },
    questions: [{ id: HOSTILE, label: HOSTILE, question: HOSTILE, response: { answer: HOSTILE, insufficient_context: false, sources: [source], removed_citations: 0 } }],
    bug_checks: [
      { id: "duplicate", label: "d", request, response: { verdict: "duplicate", duplicate_of: HOSTILE, matched_report: "BUG-0002", candidates: [candidate], routing, proposed_action: null } },
      { id: "new", label: "n", request, response: { verdict: HOSTILE, candidates: [candidate], routing, proposed_action: proposal } },
    ],
    file_anyway: { request, response: { verdict: "duplicate", candidates: [], routing, proposed_action: proposal } },
    filed_issue_example: { url: "javascript:alert(1)", title: HOSTILE },
    voice_transcript: [{ role: "agent", text: HOSTILE }, { role: "user", text: HOSTILE }],
    evals: { retrieval_hit_at_5: HOSTILE },
  };
}

function installDom(payload) {
  const nodes = new Map();
  const byId = (id) => {
    if (!nodes.has(id)) nodes.set(id, new FakeElement("div", id));
    return nodes.get(id);
  };
  const bugChips = ["duplicate", "new", "injection"].map((bug) => {
    const b = new FakeElement("button");
    b.dataset.bug = bug;
    b.disabled = true;
    return b;
  });
  const evalNodes = ["retrieval_hit_at_5", "citation_validity", "duplicates_heldout", "injection_pass", "routing_component_accuracy"].map((key) => {
    const n = new FakeElement("dd");
    n.dataset.eval = key;
    return n;
  });
  const questionButtons = () => byId("question-chips").children.map((li) => li.children[0]);
  const fetches = [];
  globalThis.document = {
    getElementById: byId,
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => ({ textContent: String(text), nodeType: 3 }),
    querySelectorAll: (selector) => {
      if (selector === "button[data-bug]") return bugChips;
      if (selector === "button[data-index]") return questionButtons();
      if (selector === "[data-eval]") return evalNodes;
      return [];
    },
    querySelector: (selector) => {
      const index = selector.match(/^button\[data-index="(\d+)"\]$/);
      if (index) return questionButtons()[Number(index[1])] ?? null;
      const bug = selector.match(/^button\[data-bug="(\w+)"\]$/);
      if (bug) return bugChips.find((b) => b.dataset.bug === bug[1]) ?? null;
      return null;
    },
  };
  globalThis.window = { matchMedia: () => ({ matches: true }) };
  globalThis.fetch = async (url, init) => {
    fetches.push({ url, init });
    return { ok: true, json: async () => payload };
  };
  byId("question-chips"); // the init guard
  return { byId, bugChips, evalNodes, questionButtons, fetches };
}

/** Every text value in a subtree (element textContent and text nodes). */
function allText(node) {
  const out = [node.textContent];
  for (const child of node.children ?? []) out.push(...allText(child));
  return out;
}

test("hostile recorded data is rendered as text only, with one same-origin fetch", async () => {
  const dom = installDom(hostileShowcase());
  await import(`${pathToFileURL(scriptPath).href}?dom-hostile`);
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
  const { byId } = dom;

  assert.equal(dom.fetches.length, 1);
  assert.equal(dom.fetches[0].url, "data/showcase.json");
  assert.equal(dom.fetches[0].init.credentials, "same-origin");
  assert.equal(dom.fetches[0].init.redirect, "error");
  assert.match(byId("load-status").textContent, /^Recorded results loaded/);

  // FakeElement throws on any innerHTML access, so reaching here means none was used.
  assert.equal(byId("answer-text").textContent, HOSTILE);
  assert.equal(byId("answer-question").textContent, `Question: ${HOSTILE}`);
  assert.ok(allText(byId("sources")).includes(HOSTILE));
  assert.equal(byId("banner-llm").textContent, HOSTILE);
  assert.notEqual(byId("count-bugs").textContent, HOSTILE, "non-integer counts keep the page default");
  assert.equal(dom.evalNodes[0].textContent, "n/a");
  assert.equal(dom.questionButtons().length, 1);
  assert.equal(dom.questionButtons()[0].textContent, HOSTILE);
  assert.equal(dom.questionButtons()[0].dataset.index, "0");
  assert.equal(dom.questionButtons()[0].id, "", "data ids never become element ids");

  // The first bug check (duplicate) is shown; hostile report text stays text.
  assert.equal(byId("report-title").textContent, HOSTILE);
  assert.equal(byId("report-description").textContent, HOSTILE);
  assert.equal(byId("routing-component").textContent, HOSTILE);
  assert.ok(allText(byId("candidates")).includes(HOSTILE));
  assert.equal(byId("file-anyway").hidden, false);

  // Bug chips: only checks present in the data are enabled.
  assert.deepEqual(dom.bugChips.map((b) => b.disabled), [false, false, true]);

  // An invalid filed-issue URL never becomes a link.
  assert.equal(byId("filed-issue-link").hidden, true);
  assert.equal(byId("filed-issue-link").href, undefined);
  assert.equal(byId("replay-filing-text").textContent, "Filing is disabled in the recorded demo.");

  // Transcript lines are text nodes.
  const lines = byId("transcript").children;
  assert.equal(lines.length, 2);
  assert.ok(lines.every((li) => li.children.some((c) => c.nodeType === 3 && c.textContent === HOSTILE)));

  // Clicking the new-bug chip shows the hostile preview as text; Confirm/Cancel stay disabled.
  dom.bugChips[1].listeners.click({ currentTarget: dom.bugChips[1] });
  assert.equal(byId("proposal").hidden, false);
  assert.equal(byId("proposal-body").textContent, HOSTILE);
  assert.equal(byId("proposal-repo").textContent, HOSTILE);
  assert.equal(byId("bug-verdict").textContent, HOSTILE);
  assert.equal(dom.bugChips[1].attributes["aria-pressed"], "true");
  assert.equal(dom.bugChips[0].attributes["aria-pressed"], "false");
  assert.ok(byId("bug-report-heading").focused >= 1, "focus moves to the result");

  // File it anyway shows the recorded step.
  dom.bugChips[0].listeners.click({ currentTarget: dom.bugChips[0] });
  byId("file-anyway-btn").listeners.click({});
  assert.equal(byId("report-file-anyway").hidden, false);
  assert.equal(byId("proposal").hidden, false);
  assert.equal(byId("file-anyway").hidden, true);

  // Still exactly one network request.
  assert.equal(dom.fetches.length, 1);
});

test("a failed load shows the error and enables nothing", async () => {
  const dom = installDom(null);
  globalThis.fetch = async (url) => {
    dom.fetches.push({ url });
    return { ok: false, json: async () => ({}) };
  };
  await import(`${pathToFileURL(scriptPath).href}?dom-fail`);
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
  assert.equal(dom.byId("load-error").hidden, false);
  assert.equal(dom.byId("load-status").textContent, "");
  assert.ok(dom.bugChips.every((b) => b.disabled));
  assert.equal(dom.questionButtons().length, 0);
});

test("an unknown schema is treated as a failed load", async () => {
  const dom = installDom({ schema: 2, questions: [{ question: "q", response: { answer: "a" } }] });
  await import(`${pathToFileURL(scriptPath).href}?dom-schema`);
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
  assert.equal(dom.byId("load-error").hidden, false);
  assert.equal(dom.questionButtons().length, 0);
});

await run();
