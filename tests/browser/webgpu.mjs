import { expect, test as base } from "./fixtures.mjs";

const modelLib =
  globalThis.process.env.WEBLLM_TEST_MODEL_LIB ??
  (globalThis.process.env.WEBLLM_TEST_MODEL_LIB_PATH
    ? "http://127.0.0.1:4178/model.wasm"
    : undefined);
export const modelId = "Qwen3-0.6B-q4f16_1-MLC";
const modelSource = globalThis.process.env.WEBLLM_TEST_MODEL_PATH
  ? "http://127.0.0.1:4178/model/"
  : undefined;

if (
  globalThis.process.env.WEBLLM_TEST_MODEL_LIB &&
  globalThis.process.env.WEBLLM_TEST_MODEL_LIB_PATH
) {
  throw new Error("Set only one model library URL or local path.");
}

export async function watchGPUErrors(page) {
  const errors = [];
  await page.exposeFunction("webllmTestGPUFailure", (message) =>
    errors.push(message),
  );
  await page.addInitScript(() => {
    let fail;
    globalThis.gpuFailure = new Promise((_, reject) => {
      fail = reject;
    });
    void globalThis.gpuFailure.catch(() => undefined);
    if (globalThis.GPUAdapter === undefined) return;
    const requestDevice = globalThis.GPUAdapter.prototype.requestDevice;
    globalThis.GPUAdapter.prototype.requestDevice = async function (...args) {
      const device = await requestDevice.apply(this, args);
      device.addEventListener(
        "uncapturederror",
        (event) => {
          void globalThis.webllmTestGPUFailure(event.error.message);
          fail(new Error(event.error.message));
        },
        { once: true },
      );
      return device;
    };
  });
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("requestfailed", (request) => {
    const url = new globalThis.URL(request.url());
    globalThis.console.error(
      url.origin + url.pathname,
      request.failure()?.errorText,
    );
  });
  return () => expect(errors).toEqual([]);
}

export const test = base.extend({
  gpuErrors: [
    async ({ page }, use) => {
      test.setTimeout(300_000);
      test.skip(
        !modelLib,
        "Set WEBLLM_TEST_MODEL_LIB or WEBLLM_TEST_MODEL_LIB_PATH to a checkpoint-capable Qwen3-0.6B WASM",
      );
      const check = await watchGPUErrors(page);
      await use();
      check();
    },
    { auto: true },
  ],
});
export { expect };

export async function loadModel(page) {
  await page.goto("/");
  await page.waitForFunction(
    () => globalThis.webllmBrowserHarness !== undefined,
  );
  await page.evaluate(
    async ({ modelId, modelLib, modelSource }) => {
      const { MLCEngine, prebuiltAppConfig } = globalThis.webllmBrowserHarness;
      const model = prebuiltAppConfig.model_list.find(
        (item) => item.model_id === modelId,
      );
      const engine = new MLCEngine({
        appConfig: {
          model_list: [
            {
              ...model,
              model: modelSource ?? model.model,
              model_lib: modelLib,
            },
          ],
        },
      });
      await Promise.race([
        engine.reload(modelId, {
          context_window_size: 512,
          prefill_chunk_size: 128,
        }),
        globalThis.gpuFailure,
      ]);
      globalThis.gpuEngine = engine;
    },
    { modelId, modelLib, modelSource },
  );
}

export function makeRequest(overrides = {}) {
  return {
    model: modelId,
    messages: [
      { role: "user", content: "List the numbers from 1 through 20." },
    ],
    seed: 17,
    temperature: 0.7,
    max_tokens: 24,
    ignore_eos: true,
    ...overrides,
  };
}

export async function baselineText(page, request) {
  return page.evaluate(async (request) => {
    const response = await Promise.race([
      globalThis.gpuEngine.chatCompletion(request),
      globalThis.gpuFailure,
    ]);
    return response.choices[0].message.content;
  }, request);
}

export async function inspectSession(page, sessionId) {
  return page.evaluate(async (sessionId) => {
    const { BrowserOPFSFileStore, ResumableSessionStore, readJournalRecords } =
      globalThis.webllmBrowserHarness;
    const files = new BrowserOPFSFileStore();
    const sessions = new ResumableSessionStore(files);
    const paths = sessions.getSessionPaths(sessionId);
    return {
      ...(await readJournalRecords(files, paths.journalPath)),
      checkpoints: await sessions.listCommittedCheckpoints(sessionId),
      directories: await files.list(paths.kvDir),
    };
  }, sessionId);
}
