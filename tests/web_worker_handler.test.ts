import { UnknownMessageKindError } from "../src/error";
import {
  CreateWebWorkerMLCEngine,
  WebWorkerMLCEngine,
  WebWorkerMLCEngineHandler,
} from "../src/web_worker";
import { jest, test, expect, beforeEach } from "@jest/globals";

const reloadMock = jest.fn<(...args: any[]) => Promise<void>>(
  async () => undefined,
);
const forwardMock = jest.fn<(...args: any[]) => Promise<any>>();
const chatCompletionMock = jest.fn<(...args: any[]) => Promise<any>>();
const completionMock = jest.fn<(...args: any[]) => Promise<any>>();
const embeddingMock = jest.fn<(...args: any[]) => Promise<any>>();
const setLogitRegistryMock = jest.fn<(...args: any[]) => void>();
const setAppConfigMock = jest.fn<(...args: any[]) => void>();

const mockEngineInstance: Record<string, any> = {
  reload: reloadMock,
  forwardTokensAndSample: forwardMock,
  chatCompletion: chatCompletionMock,
  completion: completionMock,
  embedding: embeddingMock,
  setInitProgressCallback: jest.fn((cb) => {
    mockEngineInstance.__initCb = cb;
  }),
  setLogitProcessorRegistry: setLogitRegistryMock,
  setAppConfig: setAppConfigMock,
};

jest.mock("../src/engine", () => {
  return {
    MLCEngine: jest.fn(() => mockEngineInstance),
  };
});

beforeEach(() => {
  reloadMock.mockClear();
  forwardMock.mockClear();
  chatCompletionMock.mockClear();
  completionMock.mockClear();
  embeddingMock.mockClear();
  setLogitRegistryMock.mockClear();
  setAppConfigMock.mockClear();
  mockEngineInstance.__initCb = undefined;
  (globalThis as any).postMessage = jest.fn();
});

function flushMicrotasks() {
  return new Promise<void>((resolve) => setTimeout(resolve, 0));
}

test("constructor registers init progress callback and posts updates", () => {
  const handler = new WebWorkerMLCEngineHandler();
  expect(mockEngineInstance.setInitProgressCallback).toHaveBeenCalled();
  const report = { progress: 0.5 };
  mockEngineInstance.__initCb(report);
  expect(globalThis.postMessage).toHaveBeenCalledWith({
    kind: "initProgressCallback",
    uuid: "",
    content: report,
  });
  // suppress unused
  expect(handler).toBeTruthy();
});

test("chatCompletionNonStreaming reloads when worker state mismatches", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  chatCompletionMock.mockResolvedValueOnce({ object: "chat.completion" });
  const message = {
    kind: "chatCompletionNonStreaming",
    uuid: "task-1",
    content: {
      modelId: ["demo"],
      chatOpts: [],
      request: { model: "demo", messages: [{ role: "user", content: "hi" }] },
    },
  };
  const onComplete = jest.fn();
  handler.onmessage(message, onComplete);
  await flushMicrotasks();
  expect(reloadMock).toHaveBeenCalledWith(["demo"], []);
  expect(chatCompletionMock).toHaveBeenCalled();
  expect(onComplete).toHaveBeenCalledWith({ object: "chat.completion" });
  expect(globalThis.postMessage).toHaveBeenCalledWith({
    kind: "return",
    uuid: "task-1",
    content: { object: "chat.completion" },
  });
});

test("chatCompletionStreamInit registers async generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  async function* generator() {
    yield { object: "chunk" } as any;
  }
  chatCompletionMock.mockResolvedValueOnce(generator());
  const message = {
    kind: "chatCompletionStreamInit",
    uuid: "stream",
    content: {
      modelId: ["demo"],
      selectedModelId: "demo",
      chatOpts: [],
      request: {
        model: "demo",
        messages: [{ role: "user", content: "go" }],
        stream: true,
      },
    },
  };
  handler.onmessage(message, jest.fn());
  await flushMicrotasks();
  expect(
    (handler as any).loadedModelIdToAsyncGenerator.get("demo"),
  ).toBeDefined();
});

test("completionNonStreaming routes to engine completion", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  completionMock.mockResolvedValueOnce({ object: "text_completion" });
  const message = {
    kind: "completionNonStreaming",
    uuid: "comp",
    content: {
      modelId: ["demo"],
      chatOpts: [],
      request: { model: "demo", prompt: "hi" },
    },
  };
  const onComplete = jest.fn();
  handler.onmessage(message, onComplete);
  await flushMicrotasks();
  expect(completionMock).toHaveBeenCalled();
  expect(onComplete).toHaveBeenCalledWith({ object: "text_completion" });
});

test("embedding message reloads if needed and returns embeddings", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  embeddingMock.mockResolvedValueOnce({ object: "list", data: [] });
  const message = {
    kind: "embedding",
    uuid: "embed",
    content: {
      modelId: ["demo"],
      chatOpts: [],
      request: { model: "demo", input: "text" },
    },
  };
  const onComplete = jest.fn();
  handler.onmessage(message, onComplete);
  await flushMicrotasks();
  expect(embeddingMock).toHaveBeenCalledWith({ model: "demo", input: "text" });
  expect(onComplete).toHaveBeenCalledWith({ object: "list", data: [] });
});

test("reloadIfUnmatched triggers reload when model lists differ", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  handler.modelId = ["a"];
  await handler.reloadIfUnmatched(["b"]);
  expect(reloadMock).toHaveBeenCalledWith(["b"], undefined);
  reloadMock.mockClear();
  handler.modelId = ["same"];
  await handler.reloadIfUnmatched(["same"]);
  expect(reloadMock).not.toHaveBeenCalled();
});

test("unknown messages invoke onError and throw", () => {
  const handler = new WebWorkerMLCEngineHandler();
  const onError = jest.fn();
  expect(() =>
    handler.onmessage({ kind: "mystery", content: {} }, undefined, onError),
  ).toThrow(UnknownMessageKindError);
  expect(onError).toHaveBeenCalled();
});

test("CreateWebWorkerMLCEngine instantiates client and reloads", async () => {
  const worker = { postMessage: jest.fn(), onmessage: undefined as any };
  const reloadSpy = jest
    .spyOn(WebWorkerMLCEngine.prototype, "reload")
    .mockResolvedValue(undefined);
  const engine = await CreateWebWorkerMLCEngine(worker, "model@a");
  expect(reloadSpy).toHaveBeenCalledWith("model@a", undefined);
  expect(engine.worker).toBe(worker);
  reloadSpy.mockRestore();
});

class MockWorker {
  public sent: any[] = [];
  public onmessage?: (event: any) => void;
  private responders: Map<string, (msg: any) => any> = new Map();

  constructor() {
    this.setResponder("completionNonStreaming", () => ({
      object: "completion",
    }));
    this.setResponder("embedding", () => ({ object: "list", data: [] }));
    this.setResponder("reload", () => null);
  }

  setResponder(kind: string, responder: (msg: any) => any) {
    this.responders.set(kind, responder);
  }

  postMessage = (msg: any) => {
    this.sent.push(msg);
    const responder = this.responders.get(msg.kind);
    if (!responder) {
      return;
    }
    setTimeout(async () => {
      const content = await responder(msg);
      this.onmessage?.({ kind: "return", uuid: msg.uuid, content });
    }, 0);
  };
}

test("WebWorkerMLCEngine completion sends message to worker", async () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  await engine.reload("demo-model");
  const res = await engine.completion({
    model: "demo-model",
    prompt: "hello",
  });
  expect(res.object).toBe("completion");
  expect(worker.sent.some((msg) => msg.kind === "completionNonStreaming")).toBe(
    true,
  );
});

test("WebWorkerMLCEngine embedding delegates to worker", async () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  await engine.reload("demo-model");
  const res = await engine.embedding({
    model: "demo-model",
    input: "test",
  });
  expect(res.object).toBe("list");
  expect(worker.sent.some((msg) => msg.kind === "embedding")).toBe(true);
});

test("handleTask posts throw when task rejects", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const postSpy = jest
    .spyOn(handler as any, "postMessage")
    .mockImplementation(() => undefined);
  await handler.handleTask("fail", async () => {
    throw new Error("boom");
  });
  expect(postSpy).toHaveBeenCalledWith(
    expect.objectContaining({
      kind: "throw",
      uuid: "fail",
    }),
  );
});

test("completionStreamNextChunk returns data from stored generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const generator = (async function* () {
    yield { object: "chunk" };
  })();
  (handler as any).loadedModelIdToAsyncGenerator.set("demo", generator);
  const onComplete = jest.fn();
  handler.onmessage(
    {
      kind: "completionStreamNextChunk",
      uuid: "next",
      content: { selectedModelId: "demo" },
    } as any,
    onComplete,
  );
  await flushMicrotasks();
  expect(onComplete).toHaveBeenCalledWith({ object: "chunk" });
});

test("WebWorkerMLCEngine setAppConfig posts configuration message", () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  const config = { model_list: [] } as any;
  engine.setAppConfig(config);
  const message = worker.sent.find((msg) => msg.kind === "setAppConfig");
  expect(message).toBeDefined();
  expect(message?.content).toBe(config);
});

test("WebWorkerMLCEngine setLogLevel forwards to worker", () => {
  const worker = new MockWorker();
  const engine = new WebWorkerMLCEngine(worker as any);
  engine.setLogLevel("info" as any);
  const message = worker.sent.find((msg) => msg.kind === "setLogLevel");
  expect(message?.content).toBe("info");
});

test("WebWorkerMLCEngine info helpers resolve via worker messages", async () => {
  const worker = new MockWorker();
  worker.setResponder("getMessage", () => "ready");
  worker.setResponder("runtimeStatsText", () => "stats");
  worker.setResponder("getGPUVendor", () => "MockVendor");
  worker.setResponder("getMaxStorageBufferBindingSize", () => 2048);
  worker.setResponder("interruptGenerate", () => null);
  const engine = new WebWorkerMLCEngine(worker as any);
  await engine.reload("demo");
  await expect(engine.getMessage("demo")).resolves.toBe("ready");
  await expect(engine.runtimeStatsText()).resolves.toBe("stats");
  await expect(engine.getGPUVendor()).resolves.toBe("MockVendor");
  await expect(engine.getMaxStorageBufferBindingSize()).resolves.toBe(2048);
  engine.interruptGenerate();
  await flushMicrotasks();
  expect(worker.sent.some((msg) => msg.kind === "interruptGenerate")).toBe(
    true,
  );
});
