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
const listResumableSessionsMock = jest.fn<(...args: any[]) => Promise<any>>();
const resumeChatCompletionMock = jest.fn<(...args: any[]) => Promise<any>>();
const deleteResumableSessionMock = jest.fn<(...args: any[]) => Promise<any>>();
const setLogitRegistryMock = jest.fn<(...args: any[]) => void>();
const setAppConfigMock = jest.fn<(...args: any[]) => void>();

const mockEngineInstance: Record<string, any> = {
  reload: reloadMock,
  forwardTokensAndSample: forwardMock,
  chatCompletion: chatCompletionMock,
  completion: completionMock,
  embedding: embeddingMock,
  listResumableSessions: listResumableSessionsMock,
  resumeChatCompletion: resumeChatCompletionMock,
  deleteResumableSession: deleteResumableSessionMock,
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
  listResumableSessionsMock.mockClear();
  resumeChatCompletionMock.mockClear();
  deleteResumableSessionMock.mockClear();
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
      streamId: "stream-demo",
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
    (handler as any).streamIdToAsyncGenerator.get("stream-demo"),
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

test("resumable session messages route to engine", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  listResumableSessionsMock.mockResolvedValueOnce([
    {
      sessionId: "s1",
      resumable: true,
      modelId: "demo",
      emittedTokens: 1,
      processedSeqLen: 2,
      recoveryMode: "text_only",
    },
  ]);
  resumeChatCompletionMock.mockRejectedValueOnce(new Error("unsupported"));
  deleteResumableSessionMock.mockResolvedValueOnce(undefined);
  const onComplete = jest.fn();

  handler.onmessage(
    { kind: "listResumableSessions", uuid: "list", content: null } as any,
    onComplete,
  );
  await flushMicrotasks();
  expect(listResumableSessionsMock).toHaveBeenCalled();
  expect(onComplete).toHaveBeenCalledWith([
    expect.objectContaining({ sessionId: "s1" }),
  ]);

  handler.onmessage(
    {
      kind: "deleteResumableSession",
      uuid: "delete",
      content: { sessionId: "s1" },
    } as any,
    onComplete,
  );
  await flushMicrotasks();
  expect(deleteResumableSessionMock).toHaveBeenCalledWith("s1");
  expect(onComplete).toHaveBeenCalledWith(null);

  handler.onmessage(
    {
      kind: "resumeChatCompletion",
      uuid: "resume",
      content: { sessionId: "s1", options: { continueGeneration: true } },
    } as any,
    onComplete,
  );
  await flushMicrotasks();
  expect(resumeChatCompletionMock).toHaveBeenCalledWith("s1", {
    continueGeneration: true,
  });
  expect(globalThis.postMessage).toHaveBeenCalledWith(
    expect.objectContaining({ kind: "throw", uuid: "resume" }),
  );
});

test("resumeChatCompletionStreamInit registers resumed async generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  async function* generator() {
    yield { object: "chat.completion.chunk" } as any;
  }
  resumeChatCompletionMock.mockResolvedValueOnce(generator());
  const onComplete = jest.fn();

  handler.onmessage(
    {
      kind: "resumeChatCompletionStreamInit",
      uuid: "resume-stream",
      content: {
        sessionId: "session-stream",
        streamId: "resume-stream-id",
        options: { continueGeneration: true, stream: true },
      },
    } as any,
    onComplete,
  );
  await flushMicrotasks();

  expect(resumeChatCompletionMock).toHaveBeenCalledWith("session-stream", {
    continueGeneration: true,
    stream: true,
  });
  expect(onComplete).toHaveBeenCalledWith(null);
  expect(
    (handler as any).streamIdToAsyncGenerator.get("resume-stream-id"),
  ).toBeDefined();
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

test("reloadIfUnmatched records recovered worker state", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const chatOpts = [{ temperature: 0.5 }];

  await handler.reloadIfUnmatched(["demo"], chatOpts);
  await handler.reloadIfUnmatched(["demo"], chatOpts);

  expect(reloadMock).toHaveBeenCalledTimes(1);
  expect(handler.modelId).toEqual(["demo"]);
  expect(handler.chatOpts).toBe(chatOpts);
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
    this.setResponder("listResumableSessions", () => []);
    this.setResponder("deleteResumableSession", () => null);
    this.setResponder("completionStreamReturn", () => null);
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
  (handler as any).streamIdToAsyncGenerator.set("stream-demo", generator);
  const onComplete = jest.fn();
  handler.onmessage(
    {
      kind: "completionStreamNextChunk",
      uuid: "next",
      content: { streamId: "stream-demo" },
    } as any,
    onComplete,
  );
  await flushMicrotasks();
  expect(onComplete).toHaveBeenCalledWith({ object: "chunk" });
});

test("completionStreamNextChunk removes a completed generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  const generator = (async function* () {
    yield { object: "chunk" };
  })();
  (handler as any).streamIdToAsyncGenerator.set("stream-done", generator);

  for (const uuid of ["next-one", "next-done"]) {
    handler.onmessage(
      {
        kind: "completionStreamNextChunk",
        uuid,
        content: { streamId: "stream-done" },
      } as any,
      jest.fn(),
    );
    await flushMicrotasks();
  }

  expect((handler as any).streamIdToAsyncGenerator.has("stream-done")).toBe(
    false,
  );
});

test("completionStreamReturn closes and removes a stored generator", async () => {
  const handler = new WebWorkerMLCEngineHandler();
  let closed = false;
  const generator = (async function* () {
    try {
      yield { object: "chunk" };
    } finally {
      closed = true;
    }
  })();
  await generator.next();
  (handler as any).streamIdToAsyncGenerator.set("stream-cancel", generator);

  handler.onmessage(
    {
      kind: "completionStreamReturn",
      uuid: "return",
      content: { streamId: "stream-cancel" },
    } as any,
    jest.fn(),
  );
  await flushMicrotasks();

  expect(closed).toBe(true);
  expect((handler as any).streamIdToAsyncGenerator.has("stream-cancel")).toBe(
    false,
  );
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

test("WebWorkerMLCEngine resumable helpers delegate to worker", async () => {
  const worker = new MockWorker();
  worker.setResponder("resumeChatCompletion", () => ({
    sessionId: "s1",
    recoveredText: "hello",
    emittedTokens: 1,
    processedSeqLen: 2,
    replayedTokens: 0,
    recoveryMode: "text_only",
  }));
  const engine = new WebWorkerMLCEngine(worker as any);

  await expect(engine.listResumableSessions()).resolves.toEqual([]);
  await expect(engine.resumeChatCompletion("s1")).resolves.toMatchObject({
    sessionId: "s1",
    recoveredText: "hello",
  });
  await expect(engine.deleteResumableSession("s1")).resolves.toBeUndefined();
  expect(worker.sent.some((msg) => msg.kind === "listResumableSessions")).toBe(
    true,
  );
  expect(worker.sent.some((msg) => msg.kind === "resumeChatCompletion")).toBe(
    true,
  );
  expect(worker.sent.some((msg) => msg.kind === "deleteResumableSession")).toBe(
    true,
  );
});

test("WebWorkerMLCEngine resumeChatCompletion can return resumed stream", async () => {
  const worker = new MockWorker();
  let nextCount = 0;
  worker.setResponder("resumeChatCompletionStreamInit", () => null);
  worker.setResponder("completionStreamNextChunk", () => {
    nextCount++;
    return nextCount === 1
      ? {
          object: "chat.completion.chunk",
          choices: [{ delta: { content: "x" } }],
        }
      : undefined;
  });
  const engine = new WebWorkerMLCEngine(worker as any);

  const result = await engine.resumeChatCompletion("s1", {
    continueGeneration: true,
    stream: true,
  });
  const chunks = [];
  for await (const chunk of result as AsyncIterable<any>) {
    chunks.push(chunk);
  }

  expect(chunks).toHaveLength(1);
  expect(
    worker.sent.some((msg) => msg.kind === "resumeChatCompletionStreamInit"),
  ).toBe(true);
  const init = worker.sent.find(
    (msg) => msg.kind === "resumeChatCompletionStreamInit",
  );
  expect(
    worker.sent.some(
      (msg) =>
        msg.kind === "completionStreamNextChunk" &&
        msg.content.streamId === init.content.streamId,
    ),
  ).toBe(true);
});

test("WebWorkerMLCEngine return cancels a resumed stream before first next", async () => {
  const worker = new MockWorker();
  worker.setResponder("resumeChatCompletionStreamInit", () => null);
  const engine = new WebWorkerMLCEngine(worker as any);

  const stream = (await engine.resumeChatCompletion("s-cancel", {
    continueGeneration: true,
    stream: true,
  })) as AsyncIterableIterator<any>;
  await stream.return!();

  const init = worker.sent.find(
    (msg) => msg.kind === "resumeChatCompletionStreamInit",
  );
  expect(worker.sent).toContainEqual(
    expect.objectContaining({
      kind: "completionStreamReturn",
      content: { streamId: init.content.streamId },
    }),
  );
});

test("WebWorkerMLCEngine gives concurrent streams distinct ids", async () => {
  const worker = new MockWorker();
  worker.setResponder("resumeChatCompletionStreamInit", () => null);
  const engine = new WebWorkerMLCEngine(worker as any);

  const first = (await engine.resumeChatCompletion("s-one", {
    continueGeneration: true,
    stream: true,
  })) as AsyncIterableIterator<any>;
  const second = (await engine.resumeChatCompletion("s-two", {
    continueGeneration: true,
    stream: true,
  })) as AsyncIterableIterator<any>;
  const ids = worker.sent
    .filter((msg) => msg.kind === "resumeChatCompletionStreamInit")
    .map((msg) => msg.content.streamId);

  expect(new Set(ids).size).toBe(2);
  await first.return!();
  await second.return!();
});

test.each(["chatCompletion", "completion"] as const)(
  "WebWorkerMLCEngine gives concurrent %s streams distinct IDs",
  async (endpoint) => {
    const worker = new MockWorker();
    const kind =
      endpoint === "chatCompletion"
        ? "chatCompletionStreamInit"
        : "completionStreamInit";
    worker.setResponder(kind, () => null);
    const engine = new WebWorkerMLCEngine(worker as any);
    engine.modelId = ["demo"];
    const start = () =>
      endpoint === "chatCompletion"
        ? engine.chatCompletion({
            model: "demo",
            messages: [{ role: "user", content: "hello" }],
            stream: true,
          })
        : engine.completion({ model: "demo", prompt: "hello", stream: true });
    const first = (await start()) as AsyncIterableIterator<any>;
    const second = (await start()) as AsyncIterableIterator<any>;
    const ids = worker.sent
      .filter((msg) => msg.kind === kind)
      .map((msg) => msg.content.streamId);
    expect(new Set(ids).size).toBe(2);
    await first.return!();
    await second.return!();
    expect(
      worker.sent
        .filter((msg) => msg.kind === "completionStreamReturn")
        .map((msg) => msg.content.streamId),
    ).toEqual(ids);
  },
);

test("WebWorkerMLCEngine return cancels an ordinary stream before first next", async () => {
  const worker = new MockWorker();
  worker.setResponder("chatCompletionStreamInit", () => null);
  const engine = new WebWorkerMLCEngine(worker as any);
  engine.modelId = ["demo"];

  const stream = (await engine.chatCompletion({
    model: "demo",
    messages: [{ role: "user", content: "Cancel" }],
    stream: true,
  })) as AsyncIterableIterator<any>;
  await stream.return!();

  const init = worker.sent.find(
    (msg) => msg.kind === "chatCompletionStreamInit",
  );
  expect(worker.sent).toContainEqual(
    expect.objectContaining({
      kind: "completionStreamReturn",
      content: { streamId: init.content.streamId },
    }),
  );
});

function connectWorkerStream(source: AsyncGenerator<any, void, void>) {
  const handler = new WebWorkerMLCEngineHandler();
  const worker = {
    onmessage: (_event: any) => {
      void _event;
    },
    postMessage: (msg: any) => {
      setTimeout(() => handler.onmessage(msg), 0);
    },
  };
  handler.postMessage = (msg) => worker.onmessage(msg);
  (handler as any).streamIdToAsyncGenerator.set("ordered-stream", source);
  const engine = new WebWorkerMLCEngine(worker);
  return {
    iterator: engine.asyncGenerate("ordered-stream"),
    streams: (handler as any).streamIdToAsyncGenerator as Map<string, unknown>,
  };
}

test("worker read-ahead returns done for every read beyond stream exhaustion", async () => {
  const chunk = { object: "chat.completion.chunk", choices: [] };
  const { iterator, streams } = connectWorkerStream(
    (async function* () {
      yield chunk;
    })(),
  );
  await expect(
    Promise.all([iterator.next(), iterator.next(), iterator.next()]),
  ).resolves.toEqual([
    { done: false, value: chunk },
    { done: true, value: undefined },
    { done: true, value: undefined },
  ]);
  expect(streams.size).toBe(0);
});

test("worker return waits for a pending read and closes the stream", async () => {
  let unblock!: () => void;
  const gate = new Promise<void>((resolve) => {
    unblock = resolve;
  });
  let closes = 0;
  const chunk = { object: "chat.completion.chunk", choices: [] };
  const { iterator, streams } = connectWorkerStream(
    (async function* () {
      try {
        await gate;
        yield chunk;
        yield chunk;
      } finally {
        closes++;
      }
    })(),
  );
  const pending = iterator.next();
  const returned = iterator.return();
  await flushMicrotasks();
  expect(closes).toBe(0);
  unblock();
  expect(await pending).toEqual({ done: false, value: chunk });
  expect(await returned).toEqual({ done: true, value: undefined });
  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  expect(closes).toBe(1);
  expect(streams.size).toBe(0);
});

test("worker inference failure retires the stream and queued reads return done", async () => {
  const { iterator, streams } = connectWorkerStream(
    (async function* () {
      yield { object: "chat.completion.chunk", choices: [] };
      throw new Error("decode failed");
    })(),
  );
  await iterator.next();
  const failure = expect(iterator.next()).rejects.toBe("Error: decode failed");
  const afterFailure = iterator.next();
  await failure;
  expect(await afterFailure).toEqual({ done: true, value: undefined });
  expect(streams.size).toBe(0);
});

test("worker throw before first read closes the unused stream", async () => {
  let started = false;
  const { iterator, streams } = connectWorkerStream(
    (async function* () {
      started = true;
      yield { object: "chat.completion.chunk", choices: [] };
    })(),
  );
  const failure = new Error("consumer stopped");
  await expect(iterator.throw(failure)).rejects.toBe(failure);
  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  expect(started).toBe(false);
  expect(streams.size).toBe(0);
});
