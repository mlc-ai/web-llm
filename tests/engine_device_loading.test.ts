import { afterAll, beforeEach, expect, jest, test } from "@jest/globals";
import * as tvmjs from "@mlc-ai/web-runtime";
import { MLCEngine } from "../src/engine";
import { LLMChatPipeline } from "../src/llm_chat";
import { asyncLoadTokenizer } from "../src/cache_util";
import { detectGPUDeviceForModel } from "../src/gpu_device";

jest.mock("@mlc-ai/web-runtime", () => ({
  instantiate: jest.fn(),
  createArtifactCache: jest.fn(),
  createPolyfillWASI: jest.fn(),
}));
jest.mock("../src/gpu_device", () => ({ detectGPUDeviceForModel: jest.fn() }));
jest.mock("../src/cache_util", () => ({
  ...jest.requireActual<typeof import("../src/cache_util")>(
    "../src/cache_util",
  ),
  asyncLoadTokenizer: jest.fn(),
}));
jest.mock("../src/llm_chat", () => ({ LLMChatPipeline: jest.fn() }));
const select = jest.mocked(detectGPUDeviceForModel);
const loadTokenizer = jest.mocked(asyncLoadTokenizer);
const Pipeline = jest.mocked(LLMChatPipeline);
const previousLocation = Object.getOwnPropertyDescriptor(
  globalThis,
  "location",
);
Object.defineProperty(globalThis, "location", {
  configurable: true,
  value: { origin: "https://test.example" },
});
afterAll(() => {
  if (previousLocation)
    Object.defineProperty(globalThis, "location", previousLocation);
  else Reflect.deleteProperty(globalThis, "location");
});
beforeEach(() => {
  jest.clearAllMocks();
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
async function until(called: () => boolean) {
  for (let i = 0; i < 30 && !called(); ++i) await Promise.resolve();
  expect(called()).toBe(true);
}
function fixture() {
  let initialized = false;
  const device = {
    destroy: jest.fn(),
    lost: new Promise(() => {}),
    features: new Set(["shader-f16"]),
    limits: { maxStorageBufferBindingSize: 2 ** 31 },
  } as unknown as GPUDevice;
  const selected = {
    device,
    adapter: {} as GPUAdapter,
    adapterInfo: { description: "test", vendor: "test" } as GPUAdapterInfo,
  };
  const tokenizer = { dispose: jest.fn() };
  const runtime = {
    initWebGPU: jest
      .fn<(device: GPUDevice) => void>()
      .mockImplementation(() => {
        initialized = true;
      }),
    dispose: jest.fn(() => {
      if (initialized) device.destroy();
    }),
    webgpu: jest.fn(() => ({})),
    fetchTensorCache: jest
      .fn<
        (
          url: string,
          device: unknown,
          options: { signal?: AbortSignal },
        ) => Promise<void>
      >()
      .mockResolvedValue(undefined),
  };
  const pipeline = {
    asyncLoadWebGPUPipelines: jest
      .fn<() => Promise<void>>()
      .mockResolvedValue(undefined),
    dispose: jest.fn(() => {
      runtime.dispose();
      tokenizer.dispose();
    }),
    sync: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
  };
  select.mockResolvedValue(selected);
  loadTokenizer.mockResolvedValue(
    tokenizer as unknown as Awaited<ReturnType<typeof asyncLoadTokenizer>>,
  );
  jest
    .mocked(tvmjs.instantiate)
    .mockResolvedValue(runtime as unknown as tvmjs.Instance);
  jest.mocked(tvmjs.createArtifactCache).mockReturnValue({
    fetchWithCache: jest
      .fn<() => Promise<ArrayBuffer>>()
      .mockResolvedValue(new TextEncoder().encode("{}").buffer),
  } as unknown as ReturnType<typeof tvmjs.createArtifactCache>);
  Pipeline.mockImplementation(
    () => pipeline as unknown as jest.Mocked<LLMChatPipeline>,
  );
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model_id: "test",
          model: "https://test.example/model",
          model_lib: "https://test.example/model.wasm",
          buffer_size_required_bytes: 1271398400,
        },
      ],
    },
  });
  return { engine, device, selected, tokenizer, runtime, pipeline };
}

test("cleans up a device returned after unload cancels selection", async () => {
  const f = fixture();
  const pending = deferred<tvmjs.GPUDeviceDetectOutput>();
  select.mockReturnValue(pending.promise);
  const loading = f.engine.reload("test");
  await until(() => select.mock.calls.length > 0);
  await f.engine.unload();
  pending.resolve(f.selected);
  await loading;
  expect(f.runtime.initWebGPU).not.toHaveBeenCalled();
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(Pipeline).not.toHaveBeenCalled();
});

test("cleans up runtime when device selection fails", async () => {
  const f = fixture();
  const failure = new Error("missing shader feature");
  select.mockRejectedValue(failure);
  await expect(f.engine.reload("test")).rejects.toBe(failure);
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).not.toHaveBeenCalled();
});

test("destroys the selected device if initWebGPU fails before ownership transfer", async () => {
  const f = fixture();
  const failure = new Error("partial init");
  f.runtime.initWebGPU.mockImplementation(() => {
    throw failure;
  });
  await expect(f.engine.reload("test")).rejects.toBe(failure);
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
});

test("releases the runtime-owned device when tokenizer loading fails", async () => {
  const f = fixture();
  loadTokenizer.mockRejectedValue(new Error("tokenizer failure"));
  await expect(f.engine.reload("test")).rejects.toThrow("tokenizer failure");
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
});

test("cleans up a tokenizer returned after unload cancels loading", async () => {
  const f = fixture();
  const pending = deferred<Awaited<ReturnType<typeof asyncLoadTokenizer>>>();
  loadTokenizer.mockReturnValue(pending.promise);
  const loading = f.engine.reload("test");
  await until(() => loadTokenizer.mock.calls.length > 0);
  await f.engine.unload();
  pending.resolve(
    f.tokenizer as unknown as Awaited<ReturnType<typeof asyncLoadTokenizer>>,
  );
  await loading;
  expect(f.tokenizer.dispose).toHaveBeenCalledTimes(1);
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
  expect(f.runtime.fetchTensorCache).not.toHaveBeenCalled();
});

test("cache abort uses the captured signal and releases resources once", async () => {
  const f = fixture();
  f.runtime.fetchTensorCache.mockImplementation(
    (_url, _device, options) =>
      new Promise((_resolve, reject) => {
        options.signal!.addEventListener("abort", () =>
          reject(new DOMException("abort", "AbortError")),
        );
      }),
  );
  const loading = f.engine.reload("test");
  await until(() => f.runtime.fetchTensorCache.mock.calls.length > 0);
  await f.engine.unload();
  await loading;
  expect(f.tokenizer.dispose).toHaveBeenCalledTimes(1);
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
  expect(Pipeline).not.toHaveBeenCalled();
});

test("pipeline initialization failure disposes its owned resources once", async () => {
  const f = fixture();
  f.pipeline.asyncLoadWebGPUPipelines.mockRejectedValue(
    new Error("shader failure"),
  );
  await expect(f.engine.reload("test")).rejects.toThrow("shader failure");
  expect(f.pipeline.dispose).toHaveBeenCalledTimes(1);
  expect(f.runtime.dispose).toHaveBeenCalledTimes(1);
  expect(f.tokenizer.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
});

test("success transfers ownership to the registered pipeline", async () => {
  const f = fixture();
  await f.engine.reload("test");
  expect(f.runtime.dispose).not.toHaveBeenCalled();
  expect(f.device.destroy).not.toHaveBeenCalled();
  await f.engine.unload();
  expect(f.pipeline.dispose).toHaveBeenCalledTimes(1);
  expect(f.device.destroy).toHaveBeenCalledTimes(1);
});

test("an older reload cannot clear the replacement reload's abort controller", async () => {
  const first = fixture();
  const firstSelection = deferred<tvmjs.GPUDeviceDetectOutput>();
  select.mockReturnValue(firstSelection.promise);
  const oldLoad = first.engine.reload("test");
  await until(() => select.mock.calls.length === 1);
  const second = fixture();
  const secondSelection = deferred<tvmjs.GPUDeviceDetectOutput>();
  select.mockReturnValue(secondSelection.promise);
  const newLoad = first.engine.reload("test");
  await until(() => select.mock.calls.length === 2);
  firstSelection.resolve(first.selected);
  await oldLoad;
  await first.engine.unload();
  secondSelection.resolve(second.selected);
  await newLoad;
  expect(first.device.destroy).toHaveBeenCalledTimes(1);
  expect(second.device.destroy).toHaveBeenCalledTimes(1);
  expect(second.runtime.initWebGPU).not.toHaveBeenCalled();
});
