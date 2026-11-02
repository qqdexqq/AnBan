// Drives the real UndoConfirm.tsx in JSDOM. It checks that the dialog asks
// before anything happens, that cancel paths never confirm, and that a click
// can only start one undo even before React rerenders.
// Run: node --test web/tests/undo-confirm.test.mjs
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

const entry = fileURLToPath(new URL("../src/components/UndoConfirm.tsx", import.meta.url));
const cacheFile = fileURLToPath(new URL("../node_modules/.cache/undo-confirm-test.cjs", import.meta.url));
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
const UndoConfirm = compiled.exports.default;
const act = React.act ?? ReactDOMTestUtils.act;

const RECEIPT = "Transfer of ¥50.00 to Alex — ref 8812";

let active = null;

function setup() {
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
  setGlobal("KeyboardEvent", dom.window.KeyboardEvent);

  const env = {
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
    root.render(React.createElement(UndoConfirm, props));
  });
  active.root = root;
  return container;
}

async function unmount(root) {
  await act(async () => root.unmount());
  if (active.root === root) active.root = null;
}

function button(container, pattern) {
  return [...container.querySelectorAll("button")].find((el) => pattern.test(el.textContent ?? ""));
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

async function pressEscape() {
  await act(async () => {
    globalThis.document.dispatchEvent(
      new globalThis.window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
    );
  });
}

function deferred() {
  let resolve;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

test("renders the receipt and asks before any undo, with No focused", async () => {
  setup();
  let confirms = 0;
  let cancels = 0;
  const container = await mount({
    receipt: RECEIPT,
    onConfirm: async () => {
      confirms += 1;
    },
    onCancel: () => {
      cancels += 1;
    },
  });

  assert.match(container.textContent, /Undo this transfer\?/);
  assert.match(container.textContent, /Transfer of ¥50\.00 to Alex/);
  assert.match(container.textContent, /returns the funds/i);
  assert.match(container.textContent, /eligible/i);
  assert.match(container.textContent, /may not be stopped/i);
  assert.equal(confirms, 0);
  assert.equal(cancels, 0);

  const dialog = container.querySelector('[role="alertdialog"]');
  assert.ok(dialog, "the confirmation is an alert dialog");
  assert.equal(dialog.getAttribute("aria-modal"), "true");
  assert.equal(globalThis.document.getElementById(dialog.getAttribute("aria-labelledby")).textContent, "Undo this transfer?");
  assert.match(globalThis.document.getElementById(dialog.getAttribute("aria-describedby")).textContent, /eligible/i);
  assert.equal(globalThis.document.activeElement, button(container, /No, keep transfer/i));
});

test("No cancels explicitly without confirming", async () => {
  setup();
  let confirms = 0;
  let cancels = 0;
  const container = await mount({
    receipt: RECEIPT,
    onConfirm: async () => {
      confirms += 1;
    },
    onCancel: () => {
      cancels += 1;
    },
  });

  await click(button(container, /No, keep transfer/i));
  assert.equal(cancels, 1);
  assert.equal(confirms, 0);
});

test("Escape cancels while idle but not while an undo is in flight", async () => {
  setup();
  let cancels = 0;
  const pending = deferred();
  await mount({
    receipt: RECEIPT,
    onConfirm: () => pending.promise,
    onCancel: () => {
      cancels += 1;
    },
  });

  await pressEscape();
  assert.equal(cancels, 1);

  const pending2 = deferred();
  await unmount(active.root);
  const container2 = await mount({
    receipt: RECEIPT,
    onConfirm: () => pending2.promise,
    onCancel: () => {
      cancels += 1;
    },
  });
  await click(button(container2, /Yes, undo transfer/i));
  await pressEscape();
  assert.equal(cancels, 1, "Escape is ignored while busy");
  pending2.resolve();
  pending.resolve();
});

test("Yes invokes onConfirm once even for rapid clicks, disabling both buttons", async () => {
  setup();
  const pending = deferred();
  let confirms = 0;
  const container = await mount({
    receipt: RECEIPT,
    onConfirm: () => {
      confirms += 1;
      return pending.promise;
    },
    onCancel() {},
  });

  const yes = button(container, /Yes, undo transfer/i);
  await act(async () => {
    yes.dispatchEvent(new globalThis.window.MouseEvent("click", { bubbles: true }));
    yes.dispatchEvent(new globalThis.window.MouseEvent("click", { bubbles: true }));
  });
  await flush();

  assert.equal(confirms, 1, "the in-flight ref blocks the second click");
  assert.equal(button(container, /Yes, undo transfer/i).disabled, true);
  assert.equal(button(container, /No, keep transfer/i).disabled, true);
  pending.resolve();
  await act(async () => {
    await pending.promise;
  });
  assert.equal(confirms, 1);
});

test("a failed undo shows a role=alert error and allows retry", async () => {
  setup();
  let confirms = 0;
  const container = await mount({
    receipt: RECEIPT,
    onConfirm: async () => {
      confirms += 1;
      if (confirms === 1) throw new Error("boom");
    },
    onCancel() {},
  });

  await click(button(container, /Yes, undo transfer/i));
  await flush();
  const alert = container.querySelector('[role="alert"]');
  assert.ok(alert, "the failure is announced");
  assert.match(alert.textContent, /check the latest transfer status/i);
  assert.equal(button(container, /Yes, undo transfer/i).disabled, false);
  assert.equal(button(container, /No, keep transfer/i).disabled, false);

  await click(button(container, /Yes, undo transfer/i));
  await flush();
  assert.equal(confirms, 2);
  assert.equal(container.querySelector('[role="alert"]'), null);
});

test("unmount removes the Escape listener and never cancels afterwards", async () => {
  setup();
  let cancels = 0;
  await mount({
    receipt: RECEIPT,
    onConfirm: async () => {},
    onCancel: () => {
      cancels += 1;
    },
  });

  await unmount(active.root);
  await pressEscape();
  assert.equal(cancels, 0);
});

test("renders the Chinese copy for lang=zh", async () => {
  setup();
  const container = await mount({
    lang: "zh",
    receipt: "向 Alex 转账 ¥50.00，参考号 8812",
    onConfirm: async () => {},
    onCancel() {},
  });

  assert.match(container.textContent, /确定撤销这笔转账？/);
  assert.match(container.textContent, /退回您的账户/);
  assert.ok(button(container, /不，保留转账/));
  assert.ok(button(container, /是，撤销转账/));
});
