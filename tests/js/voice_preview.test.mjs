// DOM-shim tests for the issue preview on the voice page: the pending banner, scroll and focus
// after checks, the visibility guard, the server-clock expiry timer and transcript tag stripping.
// Usage: node tests/js/voice_preview.test.mjs <path-to-voice.js>
// Prints one JSON object per test on stdout and exits 1 if any test failed. No network.

import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

import { installDom, makeRunner, reply } from "./dom_shim.mjs";

const voicePath = process.argv[2];
if (!voicePath) {
  console.error("usage: voice_preview.test.mjs <voice.js>");
  process.exit(2);
}

const { state, byId, ticks, runTimers, pendingTimers, fetchesTo, click } = installDom();
const { bugResponses, actionResponses, warnings, contextual } = state;

const ARM_MS = 3000;
const SERVER_DATE = "Mon, 12 Oct 2026 12:00:00 GMT";
const EXPIRES_AT = "2026-10-12T12:10:00Z"; // 10 minutes after SERVER_DATE
const LIFETIME_MS = 600_000;

const proposal = (id, extra = {}) => ({
  action_id: id,
  confirm_token: `tok-SECRET-${id}`,
  repo: "studio/sandbox",
  preview: { title: `Preview ${id}`, body: "body", labels: ["bug"] },
  expires_at: EXPIRES_AT,
  ...extra,
});
const bugBody = (id, overrides = {}) => ({
  verdict: "new",
  duplicate_of: null,
  matched_report: null,
  candidates: [],
  routing: { component: "save", severity: "high", labels: ["component:save"] },
  proposed_action: proposal(id),
  ...overrides,
});
const duplicateBody = () => bugBody("unused", {
  verdict: "duplicate",
  duplicate_of: "BUG-0001",
  matched_report: "BUG-0001",
  candidates: [{ doc_id: "BUG-0001", title: "Orig" }],
  proposed_action: null,
});
const disabledBody = (id) => bugBody(id, { proposed_action: proposal(id, { actions_disabled_reason: "ACTIONS_ENABLED is off" }) });
const goodBug = { title: "Cryo pod save lost", description: "Saving in a cryo pod loses progress.", platform: "pc", version: "1.0.2" };

const banner = () => byId("proposal-banner");
const previewNode = () => byId("proposal");
const tools = () => state.sessionOptions.clientTools;
const armAll = () => runTimers((t) => t.ms === ARM_MS);
const expiryTimers = () => pendingTimers().filter((t) => t.ms !== ARM_MS);
const submitForm = async () => {
  byId("bug-title").value = goodBug.title;
  byId("bug-description").value = goodBug.description;
  byId("bug-platform").value = goodBug.platform;
  byId("bug-version").value = goodBug.version;
  byId("bug-form").listeners.submit({ preventDefault() {} });
  await ticks();
};
const resetCalls = () => {
  for (const id of ["proposal", "proposal-heading", "bug-result", "bug-result-heading"]) {
    byId(id).scrollCalls.length = 0;
    byId(id).focusCalls.length = 0;
  }
};
const assertRevealedPreview = (label) => {
  assert.equal(previewNode().scrollCalls.length, 1, `${label}: preview scrolled once`);
  assert.equal(byId("proposal-heading").focusCalls.length, 1, `${label}: heading focused once`);
  assert.deepEqual(byId("proposal-heading").focusCalls[0], { preventScroll: true });
};
/** Settle any pending proposal through cancel so each test starts clean. */
const settle = async () => {
  if (!banner().hidden) {
    armAll();
    await click("cancel-btn", true);
    await ticks();
  }
  assert.equal(banner().hidden, true, "precondition: no pending proposal");
};

// The banner element starts hidden in voice.html.
banner().hidden = true;
const mod = await import(pathToFileURL(voicePath).href);
const { test, run } = makeRunner();

test("setup: start a voice session to get the real clientTools", async () => {
  await click("voice-toggle");
  await ticks();
  assert.ok(state.sessionOptions?.clientTools);
});

// --- banner -------------------------------------------------------------------------
test("banner shows after an agent check with a ready proposal and hides on confirm", async () => {
  bugResponses.push(bugBody("a1"));
  await tools().check_bug_report(goodBug);
  assert.equal(banner().hidden, false);
  assert.ok(previewNode().classes.has("is-pending"));
  armAll();
  await click("confirm-btn", true);
  await ticks();
  assert.equal(banner().hidden, true);
  assert.ok(!previewNode().classes.has("is-pending"));
});

test("banner shows after a form check with a ready proposal and hides on cancel", async () => {
  bugResponses.push(bugBody("f1"));
  await submitForm();
  assert.equal(banner().hidden, false);
  armAll();
  await click("cancel-btn", true);
  await ticks();
  assert.equal(banner().hidden, true);
});

test("banner stays hidden for duplicate and disabled results (agent and form)", async () => {
  for (const body of [duplicateBody(), disabledBody("d1")]) {
    bugResponses.push(body);
    await tools().check_bug_report(goodBug);
    assert.equal(banner().hidden, true, `agent ${body.verdict}`);
    bugResponses.push(body);
    await submitForm();
    assert.equal(banner().hidden, true, `form ${body.verdict}`);
  }
});

for (const status of [404, 409, 410]) {
  test(`banner hides when confirm returns ${status}`, async () => {
    await settle();
    bugResponses.push(bugBody(`c${status}`));
    await tools().check_bug_report(goodBug);
    assert.equal(banner().hidden, false);
    armAll();
    actionResponses.push(reply({ detail: "x" }, { status }));
    await click("confirm-btn", true);
    await ticks();
    assert.equal(banner().hidden, true);
  });

  test(`banner hides when cancel returns ${status}`, async () => {
    await settle();
    bugResponses.push(bugBody(`k${status}`));
    await submitForm();
    assert.equal(banner().hidden, false);
    actionResponses.push(reply({ detail: "x" }, { status }));
    await click("cancel-btn", true);
    await ticks();
    assert.equal(banner().hidden, true);
  });
}

test("banner stays shown when confirm fails with a retryable 502", async () => {
  await settle();
  bugResponses.push(bugBody("r502"));
  await submitForm();
  armAll();
  actionResponses.push(reply({ detail: "x" }, { status: 502 }));
  await click("confirm-btn", true);
  await ticks();
  assert.equal(banner().hidden, false);
  assert.equal(byId("confirm-btn").disabled, false);
  await settle();
});

// --- scroll and focus ---------------------------------------------------------------
test("a check with a ready proposal scrolls to the preview and focuses its heading", async () => {
  await settle();
  resetCalls();
  bugResponses.push(bugBody("s1"));
  await tools().check_bug_report(goodBug);
  assertRevealedPreview("agent check");
  assert.deepEqual(previewNode().scrollCalls[0], { block: "start", behavior: "smooth" });
  await settle();
  resetCalls();
  bugResponses.push(bugBody("s2"));
  await submitForm();
  assertRevealedPreview("form check");
});

test("a pending second agent call scrolls and focuses the preview again without fetching", async () => {
  resetCalls();
  const before = fetchesTo("/bugs/check").length;
  assert.equal(await tools().check_bug_report(goodBug), mod.toAgentPendingResult());
  assertRevealedPreview("pending call");
  assert.equal(fetchesTo("/bugs/check").length, before);
});

test("the banner button reveals the preview", async () => {
  resetCalls();
  byId("proposal-banner-btn").listeners.click({ isTrusted: true });
  assertRevealedPreview("banner button");
  await settle();
});

test("a duplicate verdict scrolls to the result and focuses the verdict heading", async () => {
  resetCalls();
  bugResponses.push(duplicateBody());
  await tools().check_bug_report(goodBug);
  assert.equal(previewNode().hidden, true);
  assert.equal(byId("bug-result").scrollCalls.length, 1);
  assert.equal(byId("bug-result-heading").focusCalls.length, 1);
  assert.equal(byId("proposal-heading").focusCalls.length, 0);
});

test("reduced motion scrolls without smooth behaviour", async () => {
  window.matchMedia = (query) => ({ matches: query === "(prefers-reduced-motion: reduce)" });
  resetCalls();
  bugResponses.push(bugBody("m1"));
  await tools().check_bug_report(goodBug);
  assert.deepEqual(previewNode().scrollCalls[0], { block: "start", behavior: "auto" });
  delete window.matchMedia;
  await settle();
});

// --- visibility guard ---------------------------------------------------------------
const GUARD_WARNING = ["StudioDesk: proposal pending but not visible"];

test("no guard warning when the pending preview is visible", async () => {
  warnings.length = 0;
  bugResponses.push(bugBody("g0"));
  await tools().check_bug_report(goodBug);
  await tools().check_bug_report(goodBug);
  assert.deepEqual(warnings, []);
});

for (const [label, hide] of [
  ["#proposal hidden", () => { previewNode().hidden = true; }],
  ["#bug-result hidden", () => { byId("bug-result").hidden = true; }],
  ["#bug-panel hidden", () => { byId("bug-panel").hidden = true; }],
  ["zero height", () => { previewNode().rect = { height: 0 }; }],
]) {
  test(`guard warns and forces the preview visible when ${label}`, async () => {
    warnings.length = 0;
    hide();
    assert.equal(await tools().check_bug_report(goodBug), mod.toAgentPendingResult());
    assert.deepEqual(warnings, [GUARD_WARNING]);
    assert.equal(previewNode().hidden, false);
    assert.equal(byId("bug-result").hidden, false);
    assert.equal(byId("bug-panel").hidden, false);
    assert.ok(previewNode().classes.has("is-forced-visible"));
    previewNode().rect = null;
  });
}

test("a new render clears the forced-visible marker", async () => {
  await settle();
  bugResponses.push(bugBody("g1"));
  await submitForm();
  assert.ok(!previewNode().classes.has("is-forced-visible"));
  await settle();
});

// --- expiry timer -------------------------------------------------------------------
test("without a Date header only the arm timer is scheduled", async () => {
  await settle();
  const before = state.timers.length;
  bugResponses.push(bugBody("n1"));
  await tools().check_bug_report(goodBug);
  assert.deepEqual(state.timers.slice(before).map((t) => t.ms), [ARM_MS]);
  await settle();
});

test("a Date header schedules the expiry timer from expires_at minus Date", async () => {
  const before = state.timers.length;
  bugResponses.push(reply(bugBody("e1"), { date: SERVER_DATE }));
  await tools().check_bug_report(goodBug);
  assert.deepEqual(state.timers.slice(before).map((t) => t.ms), [ARM_MS, LIFETIME_MS]);
});

test("expiry retires the current idle proposal", async () => {
  contextual.length = 0;
  runTimers((t) => t.ms === LIFETIME_MS);
  assert.equal(banner().hidden, true);
  assert.equal(byId("confirm-btn").disabled, true);
  assert.equal(byId("cancel-btn").disabled, true);
  assert.match(byId("action-status").textContent, /expired\. Nothing was filed/);
  assert.ok(contextual.some((t) => /expired/.test(t)));
  // Retired: confirm sends nothing, and the agent may check again.
  const before = state.fetches.length;
  armAll();
  await click("confirm-btn", true);
  await ticks();
  assert.equal(state.fetches.length, before);
  bugResponses.push(bugBody("e2"));
  assert.equal(JSON.parse(await tools().check_bug_report(goodBug)).proposal_ready, true);
  await settle();
});

test("an expired-on-arrival proposal gets a 0 ms timer", async () => {
  const before = state.timers.length;
  bugResponses.push(reply(bugBody("e0", { proposed_action: proposal("e0", { expires_at: "2026-10-12T11:00:00Z" }) }), { date: SERVER_DATE }));
  await tools().check_bug_report(goodBug);
  assert.deepEqual(state.timers.slice(before).map((t) => t.ms), [ARM_MS, 0]);
  runTimers((t) => t.ms === 0);
  assert.equal(banner().hidden, true);
});

test("a stale expiry timer does not retire a newer proposal", async () => {
  await settle();
  bugResponses.push(reply(bugBody("old"), { date: SERVER_DATE }));
  await submitForm();
  const [oldTimer] = expiryTimers();
  bugResponses.push(bugBody("new")); // replaces "old" via the form, no Date header
  await submitForm();
  oldTimer.fired = true;
  oldTimer.fn();
  assert.equal(banner().hidden, false);
  assert.doesNotMatch(byId("action-status").textContent, /expired/);
  armAll();
  await click("confirm-btn", true);
  await ticks();
  assert.equal(state.fetches.at(-1).path, "/actions/new/confirm");
});

test("expiry does not retire a proposal whose confirm is in flight", async () => {
  bugResponses.push(reply(bugBody("busy"), { date: SERVER_DATE }));
  await submitForm();
  const [timer] = expiryTimers();
  armAll();
  let resolveConfirm;
  actionResponses.push(new Promise((resolve) => { resolveConfirm = resolve; }));
  const confirming = click("confirm-btn", true); // not awaited: its fetch waits on resolveConfirm
  await ticks();
  timer.fired = true;
  timer.fn();
  assert.doesNotMatch(byId("action-status").textContent, /expired/);
  assert.equal(banner().hidden, false);
  resolveConfirm(reply({ issue_number: 9, issue_url: "https://github.com/studio/sandbox/issues/9", slack: "sent" }));
  await confirming;
  await ticks();
  assert.match(byId("action-status").textContent, /Issue #9 was filed/);
});

test("expiry after cancel leaves the cancelled status alone", async () => {
  bugResponses.push(reply(bugBody("gone"), { date: SERVER_DATE }));
  await submitForm();
  const [timer] = expiryTimers();
  armAll();
  await click("cancel-btn", true);
  await ticks();
  timer.fired = true;
  timer.fn();
  assert.match(byId("action-status").textContent, /Cancelled\. Nothing was filed/);
});

// --- file it anyway (M6) ----------------------------------------------------------
const fileAnywayDuplicate = () => ({ ...duplicateBody(), file_anyway_available: true });

test("a duplicate with file_anyway_available shows File it anyway; others hide it", async () => {
  await settle();
  bugResponses.push(fileAnywayDuplicate());
  await submitForm();
  assert.equal(byId("file-anyway").hidden, false);
  assert.equal(byId("file-anyway-btn").disabled, false);
  assert.match(byId("bug-status").textContent, /still file it/);
  bugResponses.push(duplicateBody());
  await submitForm();
  assert.equal(byId("file-anyway").hidden, true, "no flag: hidden");
  bugResponses.push(bugBody("fa0"));
  await submitForm();
  assert.equal(byId("file-anyway").hidden, true, "ready proposal: hidden");
  await settle();
});

test("File it anyway re-checks the judged report with file_anyway and still needs Confirm", async () => {
  await settle();
  bugResponses.push(fileAnywayDuplicate());
  await submitForm();
  // The user edits the form afterwards: the re-check must use the report that was judged.
  byId("bug-title").value = "Something else entirely";
  const before = fetchesTo("/bugs/check").length;
  await byId("file-anyway-btn").listeners.click({ isTrusted: false });
  await ticks();
  assert.equal(fetchesTo("/bugs/check").length, before, "untrusted click ignored");

  const labelled = bugBody("fa1");
  labelled.verdict = "duplicate";
  labelled.duplicate_of = "BUG-0001";
  labelled.proposed_action.preview.labels = ["bug", "possible-duplicate"];
  bugResponses.push(labelled);
  await byId("file-anyway-btn").listeners.click({ isTrusted: true });
  await ticks();
  assert.deepEqual(fetchesTo("/bugs/check").at(-1).body, { ...goodBug, file_anyway: true });
  assert.equal(byId("file-anyway").hidden, true);
  assert.equal(byId("proposal-title").textContent, "Preview fa1");
  assert.deepEqual(byId("proposal-labels").children.map((c) => c.textContent), ["bug", "possible-duplicate"]);
  assert.match(byId("bug-status").textContent, /possible-duplicate/);
  assert.equal(byId("confirm-btn").disabled, true, "confirm starts disarmed");
  assert.equal(fetchesTo("/actions/fa1/confirm").length, 0, "nothing filed without Confirm");
  assert.equal(banner().hidden, false);
  await settle();
});

test("a failed File it anyway re-enables the button and files nothing", async () => {
  await settle();
  bugResponses.push(fileAnywayDuplicate());
  await submitForm();
  bugResponses.push(reply({ detail: "boom" }, { status: 503 }));
  await byId("file-anyway-btn").listeners.click({ isTrusted: true });
  await ticks();
  assert.equal(byId("file-anyway-btn").disabled, false);
  assert.equal(byId("file-anyway").hidden, false);
  assert.match(byId("bug-status").textContent, /temporarily unavailable/);
  assert.equal(banner().hidden, true);
});

// --- transcript ---------------------------------------------------------------------
const transcriptLines = () => byId("transcript").children.map((li) => li.children.map((c) => c.textContent));

test("agent transcript lines have audio tags stripped; user lines are kept as said", () => {
  byId("transcript").replaceChildren();
  const { onMessage } = state.sessionOptions;
  onMessage({ source: "ai", message: "[happy] Great! If you have a bug" });
  onMessage({ source: "ai", message: "[happy]  " });
  onMessage({ source: "ai", message: "That is [BR-0001] on [PS5]." });
  onMessage({ source: "user", message: "[happy] my game [crashed] on load" });
  onMessage({ role: "agent", message: "[slow] Okay  [calm] noted" });
  assert.deepEqual(transcriptLines(), [
    ["Agent", "Great! If you have a bug"],
    ["Agent", "That is [BR-0001] on [PS5]."],
    ["You", "[happy] my game [crashed] on load"],
    ["Agent", "Okay noted"],
  ]);
});

await run();
