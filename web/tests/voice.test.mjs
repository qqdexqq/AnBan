// Drives the real VoiceInput.tsx in JSDOM with a fake MediaRecorder.
// Run: node --test web/tests/voice.test.mjs
import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import Module, { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const webRoot = fileURLToPath(new URL("..", import.meta.url));
const webRequire = createRequire(new URL("../package.json", import.meta.url));

const React = webRequire("react");
const { createRoot } = webRequire("react-dom/client");
const ReactDOMTestUtils = webRequire("react-dom/test-utils");
const esbuild = webRequire("esbuild");
const { JSDOM } = webRequire("jsdom");

const entry = fileURLToPath(new URL("../src/components/VoiceInput.tsx", import.meta.url));
const cacheFile = fileURLToPath(new URL("../node_modules/.cache/voice-test.cjs", import.meta.url));
const output = esbuild.buildSync({
  entryPoints: [entry],
  bundle: true,
  write: false,
  platform: "node",
  format: "cjs",
  jsx: "automatic",
  external: ["react", "react/jsx-runtime", "react-dom", "react-dom/client", "react-dom/test-utils"],
  logLevel: "silent",
});
const compiled = new Module(cacheFile);
compiled.filename = cacheFile;
compiled.paths = Module._nodeModulePaths(webRoot);
compiled._compile(output.outputFiles[0].text, cacheFile);
const VoiceInput = compiled.exports.default;
const act = React.act ?? ReactDOMTestUtils.act;

const RECORDER_MIME = "audio/webm;codecs=opus";

class FakeMediaRecorder {
  static instances = [];
  static isTypeSupported(type) {
    return type === RECORDER_MIME;
  }
  constructor(stream, options = {}) {
    this.stream = stream;
    this.mimeType = options.mimeType || "";
    this.state = "inactive";
    this.ondataavailable = null;
    this.onstop = null;
    this.onerror = null;
    FakeMediaRecorder.instances.push(this);
  }
  start() {
    this.state = "recording";
  }
  stop() {
    if (this.state === "inactive") throw new Error("InvalidStateError");
    this.state = "inactive";
    this.ondataavailable?.({ data: new globalThis.Blob(["recorded-audio"], { type: this.mimeType }) });
    this.onstop?.();
  }
}

let active = null;

function setup({ getUserMedia, fetch: fetchImpl } = {}) {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://localhost/",
    pretendToBeVisual: true,
  });
  const saved = new Map();
  const setGlobal = (name, value) => {
    if (!saved.has(name)) saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  };

  setGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  setGlobal("window", dom.window);
  setGlobal("document", dom.window.document);
  setGlobal("navigator", dom.window.navigator);
  setGlobal("HTMLElement", dom.window.HTMLElement);
  setGlobal("Node", dom.window.Node);
  setGlobal("Event", dom.window.Event);
  setGlobal("MouseEvent", dom.window.MouseEvent);
  setGlobal("Blob", dom.window.Blob);
  setGlobal("AbortController", dom.window.AbortController || AbortController);
  FakeMediaRecorder.instances = [];
  setGlobal("MediaRecorder", FakeMediaRecorder);
  Object.defineProperty(dom.window, "MediaRecorder", { configurable: true, value: FakeMediaRecorder });
  Object.defineProperty(dom.window, "isSecureContext", { configurable: true, value: true });

  Object.defineProperty(dom.window.navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia:
        getUserMedia ??
        (() => Promise.reject(Object.assign(new Error("denied"), { name: "NotAllowedError" }))),
    },
  });

  const calls = [];
  setGlobal("fetch", (url, init = {}) => {
    calls.push({ url: String(url), init });
    if (!fetchImpl) throw new Error(`unexpected fetch: ${url}`);
    return fetchImpl(String(url), init);
  });

  const env = {
    dom,
    calls,
    restore() {
      for (const [name, descriptor] of saved) {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
      }
      dom.window.close();
    },
  };
  active = { env, root: null };
  return env;
}

afterEach(async () => {
  const current = active;
  active = null;
  if (!current) return;
  if (current.root) {
    try {
      await act(async () => current.root.unmount());
    } catch {
      // Already unmounted by the test.
    }
  }
  current.env.restore();
});

async function mount(props) {
  const container = globalThis.document.createElement("div");
  globalThis.document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(VoiceInput, props));
  });
  active.root = root;
  return container;
}

async function unmount(root) {
  await act(async () => root.unmount());
  if (active.root === root) active.root = null;
}

async function click(element) {
  await act(async () => {
    element.dispatchEvent(new globalThis.window.MouseEvent("click", { bubbles: true }));
  });
}

async function flush() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function mic(container) {
  return container.querySelector(
    'button[aria-label="Start voice input"], button[aria-label="Stop recording"], button[aria-label="Transcribing voice input"]',
  );
}

function jsonResponse(body, ok = true, status = 200) {
  return { ok, status, json: async () => body, text: async () => JSON.stringify(body) };
}

function voiceFetch(text) {
  return async (url) => {
    if (url !== "/api/voice/transcribe") throw new Error(`unexpected request: ${url}`);
    return jsonResponse({ text, language: "en", model: "test" });
  };
}

function makeStream() {
  const track = {
    stopped: false,
    stop() {
      this.stopped = true;
    },
  };
  return { tracks: [track], stream: { getTracks: () => [track] } };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

test("record then stop yields the editable transcript, one raw upload and stopped tracks", async () => {
  const media = makeStream();
  const uploads = [];
  setup({
    getUserMedia: async () => media.stream,
    fetch: (url, init) => {
      uploads.push({ url, init });
      return voiceFetch("  hello world  ")(url);
    },
  });
  const transcripts = [];
  const busy = [];
  const container = await mount({
    lang: "en",
    disabled: false,
    onTranscript: (text) => transcripts.push(text),
    onBusyChange: (value) => busy.push(value),
  });

  await click(mic(container));
  await flush();
  assert.equal(FakeMediaRecorder.instances.length, 1);
  assert.equal(mic(container).getAttribute("aria-label"), "Stop recording");

  await click(mic(container));
  await flush();

  assert.deepEqual(transcripts, ["hello world"]);
  assert.equal(uploads.length, 1);
  assert.equal(uploads[0].url, "/api/voice/transcribe");
  assert.equal(uploads[0].init.method, "POST");
  assert.ok(uploads[0].init.body instanceof globalThis.Blob);
  assert.ok(uploads[0].init.body.size > 0);
  assert.equal(uploads[0].init.body.type, RECORDER_MIME);
  assert.equal(uploads[0].init.headers["Content-Type"], RECORDER_MIME);
  assert.ok(media.tracks.every((track) => track.stopped));
  assert.ok(!uploads.some((call) => /chat|approve/.test(call.url)));
  assert.ok(busy.includes(true));
  assert.equal(busy.at(-1), false);
  assert.match(container.textContent, /Transcript ready/);
});

test("permission denial shows a helpful error and clears busy", async () => {
  const env = setup({
    getUserMedia: async () => {
      throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
    },
  });
  const busy = [];
  const container = await mount({ lang: "en", disabled: false, onTranscript() {}, onBusyChange: (v) => busy.push(v) });

  await click(mic(container));
  await flush();

  const alert = container.querySelector('[role="alert"]');
  assert.ok(alert, "permission error is announced");
  assert.match(alert.textContent, /Microphone access was denied/);
  assert.equal(busy.at(-1) ?? false, false);
  assert.equal(env.calls.length, 0);
  assert.equal(mic(container).disabled, false);
  assert.ok(container.querySelector("button.pill-btn"), "retry action is offered");
});

test("unmount before permission resolves stops late tracks and never uploads", async () => {
  const pending = deferred();
  const media = makeStream();
  const env = setup({ getUserMedia: () => pending.promise });
  const transcripts = [];
  const container = await mount({ lang: "en", disabled: false, onTranscript: (t) => transcripts.push(t) });

  await click(mic(container));
  await flush();
  assert.equal(mic(container).getAttribute("aria-label"), "Stop recording");

  await unmount(active.root);
  pending.resolve(media.stream);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.ok(media.tracks.every((track) => track.stopped));
  assert.equal(FakeMediaRecorder.instances.length, 0);
  assert.equal(env.calls.length, 0);
  assert.deepEqual(transcripts, []);
});

test("unmount during an in-flight upload aborts the signal and ignores the late result", async () => {
  const media = makeStream();
  const pending = deferred();
  let signal = null;
  setup({
    getUserMedia: async () => media.stream,
    fetch: (url, init) => {
      assert.equal(url, "/api/voice/transcribe");
      signal = init.signal;
      return new Promise((resolve, reject) => {
        pending.resolve = () => resolve(jsonResponse({ text: "late transcript" }));
        signal.addEventListener("abort", () =>
          reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
        );
      });
    },
  });
  const transcripts = [];
  const container = await mount({ lang: "en", disabled: false, onTranscript: (t) => transcripts.push(t) });

  await click(mic(container));
  await flush();
  await click(mic(container));
  await flush();
  assert.ok(signal, "transcription request started");

  await unmount(active.root);
  assert.equal(signal.aborted, true);
  pending.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(transcripts, []);
});

test("disabling while recording cancels the recording without uploading", async () => {
  const media = makeStream();
  const env = setup({ getUserMedia: async () => media.stream, fetch: voiceFetch("nope") });
  const transcripts = [];
  const props = { lang: "en", disabled: false, onTranscript: (t) => transcripts.push(t) };
  const container = await mount(props);

  await click(mic(container));
  await flush();
  const recorder = FakeMediaRecorder.instances[0];
  assert.equal(recorder.state, "recording");

  await act(async () => active.root.render(React.createElement(VoiceInput, { ...props, disabled: true })));
  await flush();

  assert.equal(recorder.state, "inactive");
  assert.ok(media.tracks.every((track) => track.stopped));
  assert.deepEqual(transcripts, []);
  assert.equal(env.calls.length, 0);
  assert.equal(mic(container).disabled, true);
});
