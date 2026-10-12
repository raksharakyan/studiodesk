// DOM-shim tests for "greet once per tab" on the voice page: buildSessionOverrides, the
// sessionStorage flag (with an in-memory fallback), the silent-start hint and no auto-restart.
// Usage: node tests/js/voice_greeting.test.mjs <path-to-voice.js>
// Each scenario imports a fresh copy of voice.js (cache-busting query) so the module-level
// greeted flag starts clean; the newest copy's init() owns the fake page's listeners.
// Prints one JSON object per test on stdout and exits 1 if any test failed. No network.

import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

import { installDom, makeRunner } from "./dom_shim.mjs";

const voicePath = process.argv[2];
if (!voicePath) {
  console.error("usage: voice_greeting.test.mjs <voice.js>");
  process.exit(2);
}

const { state, byId, ticks, runTimers, click } = installDom();
const baseUrl = pathToFileURL(voicePath).href;
const KEY = "studiodesk.voice.greeted";
const SILENT = { agent: { firstMessage: "" } };
let loads = 0;

/** A Storage-like object backed by a Map; `throwOn` names methods that throw. */
function fakeStorage(initial = {}, throwOn = []) {
  const data = new Map(Object.entries(initial));
  const guard = (name) => { if (throwOn.includes(name)) throw new Error(`${name} blocked`); };
  return {
    data,
    getItem: (key) => { guard("getItem"); return data.has(key) ? data.get(key) : null; },
    setItem: (key, value) => { guard("setItem"); data.set(key, String(value)); },
    removeItem: (key) => data.delete(key),
  };
}

/** Install `storage` as window.sessionStorage: an object, undefined, or "throwing-getter". */
function setStorage(storage) {
  delete window.sessionStorage;
  if (storage === "throwing-getter") {
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get() { throw new Error("SecurityError: storage disabled"); },
    });
  } else if (storage !== undefined) {
    window.sessionStorage = storage;
  }
}

/** Load a fresh voice.js (new module state) and reset the recorded calls. */
async function freshPage(storage) {
  setStorage(storage);
  state.startCalls.length = 0;
  state.sessionOptions = null;
  state.startSessionImpl = async (options) => {
    state.sessionOptions = options;
    return { endSession: async () => {}, sendContextualUpdate() {} };
  };
  byId("voice-hint").textContent = "";
  loads += 1;
  return import(`${baseUrl}?scenario=${loads}`);
}

const start = async () => { await click("voice-toggle"); await ticks(); return state.startCalls.at(-1); };
const stop = async () => { await click("voice-toggle"); await ticks(); };
const hint = () => byId("voice-hint").textContent;

const mod = await freshPage(undefined);
const { test, run } = makeRunner();

// --- buildSessionOverrides ----------------------------------------------------------
test("buildSessionOverrides(true) is the empty first message", () => {
  assert.deepEqual(mod.buildSessionOverrides(true), SILENT);
});

test("buildSessionOverrides is {} for anything but true", () => {
  for (const value of [false, undefined, null, "1", 1, "true", {}, []]) {
    assert.deepEqual(mod.buildSessionOverrides(value), {}, String(value));
  }
});

test("buildSessionOverrides returns a fresh object each call", () => {
  const a = mod.buildSessionOverrides(true);
  const b = mod.buildSessionOverrides(true);
  assert.notEqual(a, b);
  assert.notEqual(a.agent, b.agent);
  a.agent.firstMessage = "Hacked greeting";
  assert.deepEqual(mod.buildSessionOverrides(true), SILENT);
  assert.notEqual(mod.buildSessionOverrides(false), mod.buildSessionOverrides(false));
});

test("exported constants", () => {
  assert.equal(mod.GREETED_STORAGE_KEY, KEY);
  assert.equal(mod.SILENT_START_HINT, "Listening — go ahead and speak.");
});

// --- greet once ---------------------------------------------------------------------
test("first start greets (no overrides key); after onConnect the next start is silent", async () => {
  await freshPage(undefined);
  const first = await start();
  assert.equal("overrides" in first, false);
  first.onConnect();
  assert.equal(hint(), "", "no hint when the greeting plays");
  await stop();
  const second = await start();
  assert.deepEqual(second.overrides, SILENT);
  assert.equal(state.startCalls.length, 2);
});

test("a later start stays silent on every call", async () => {
  state.startCalls.at(-1).onConnect();
  await stop();
  const third = await start();
  assert.deepEqual(third.overrides, SILENT);
});

test("onConnect stores the flag in sessionStorage", async () => {
  const storage = fakeStorage();
  await freshPage(storage);
  const first = await start();
  assert.equal("overrides" in first, false);
  assert.equal(storage.data.get(KEY), undefined, "not set before connect");
  first.onConnect();
  assert.equal(storage.data.get(KEY), "1");
  assert.deepEqual([...storage.data.keys()], [KEY], "storage holds only the greeted flag");
});

test("storage that already holds '1' gives a silent first start", async () => {
  await freshPage(fakeStorage({ [KEY]: "1" }));
  const first = await start();
  assert.deepEqual(first.overrides, SILENT);
});

test("storage holding another value still greets", async () => {
  for (const value of ["true", "0", "yes", ""]) {
    await freshPage(fakeStorage({ [KEY]: value }));
    assert.equal("overrides" in (await start()), false, value);
  }
});

for (const [label, storage] of [
  ["a throwing sessionStorage getter", () => "throwing-getter"],
  ["getItem and setItem that throw", () => fakeStorage({}, ["getItem", "setItem"])],
  ["setItem that throws", () => fakeStorage({}, ["setItem"])],
  ["getItem that throws", () => fakeStorage({ [KEY]: "1" }, ["getItem"])],
]) {
  test(`${label} still greets once via memory`, async () => {
    await freshPage(storage());
    const first = await start();
    assert.equal("overrides" in first, false);
    first.onConnect();
    await stop();
    assert.deepEqual((await start()).overrides, SILENT);
  });
}

test("if startSession rejects, the flag is not set and the next start greets", async () => {
  const storage = fakeStorage();
  await freshPage(storage);
  state.startSessionImpl = async () => { throw new Error("network down"); };
  await start();
  assert.equal(storage.data.has(KEY), false);
  state.startSessionImpl = async (options) => ({ endSession: async () => {}, options });
  const next = await start();
  assert.equal("overrides" in next, false);
  assert.equal(state.startCalls.length, 2);
});

test("if onConnect never fires, the flag is not set and the next start greets", async () => {
  const storage = fakeStorage();
  await freshPage(storage);
  await start(); // resolves, but the fake never calls onConnect
  await stop();
  assert.equal(storage.data.has(KEY), false);
  assert.equal("overrides" in (await start()), false);
});

// --- hint ---------------------------------------------------------------------------
test("the hint shows only on a silent start, after connect", async () => {
  await freshPage(fakeStorage({ [KEY]: "1" }));
  const options = await start();
  assert.equal(hint(), "", "not before connect");
  options.onConnect();
  assert.equal(hint(), "Listening — go ahead and speak.");
});

test("an empty or tag-only transcript line keeps the hint; the first real line clears it", async () => {
  const options = state.startCalls.at(-1);
  options.onMessage({ source: "ai", message: "[happy]" });
  options.onMessage({ source: "user", message: "   " });
  assert.equal(hint(), "Listening — go ahead and speak.");
  options.onMessage({ source: "user", message: "Hi, my save is gone" });
  assert.equal(hint(), "");
});

test("the hint is cleared on disconnect", async () => {
  await freshPage(fakeStorage({ [KEY]: "1" }));
  const options = await start();
  options.onConnect();
  assert.notEqual(hint(), "");
  options.onDisconnect({ reason: "user" });
  assert.equal(hint(), "");
});

test("the hint is cleared when the user stops voice", async () => {
  await freshPage(fakeStorage({ [KEY]: "1" }));
  (await start()).onConnect();
  await stop();
  assert.equal(hint(), "");
});

// --- no auto-restart ----------------------------------------------------------------
for (const reason of ["agent", "error", "user"]) {
  test(`onDisconnect({reason: "${reason}"}) never restarts the session`, async () => {
    await freshPage(fakeStorage());
    const options = await start();
    options.onConnect();
    options.onDisconnect({ reason });
    runTimers();
    await ticks(40);
    runTimers();
    await ticks(40);
    assert.equal(state.startCalls.length, 1);
    assert.equal(byId("voice-toggle").attributes["aria-pressed"], "false");
  });
}

test("after a disconnect only a Start click starts again, silently", async () => {
  await freshPage(fakeStorage());
  const options = await start();
  options.onConnect();
  options.onDisconnect({ reason: "agent" });
  await ticks();
  const next = await start();
  assert.equal(state.startCalls.length, 2);
  assert.deepEqual(next.overrides, SILENT);
});

await run();
