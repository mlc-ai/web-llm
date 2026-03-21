import {
  CreateServiceWorkerMLCEngine,
  ServiceWorker,
  ServiceWorkerMLCEngine,
  ServiceWorkerMLCEngineHandler,
} from "../src/service_worker";
import { jest, test, expect, afterEach } from "@jest/globals";

type ServiceWorkerHandlerEvent = Parameters<
  ServiceWorkerMLCEngineHandler["onmessage"]
>[0];

jest.mock("@mlc-ai/web-runtime", () => ({
  detectGPUDevice: jest.fn(async () => ({
    adapterInfo: { description: "MockGPU", vendor: "MockVendor" },
  })),
}));

const reloadMock = jest.fn();
const initCallback = jest.fn();

jest.mock("../src/engine", () => {
  return {
    MLCEngine: jest.fn(() => ({
      reload: reloadMock,
      chatCompletion: jest.fn(),
      completion: jest.fn(),
      embedding: jest.fn(),
      forwardTokensAndSample: jest.fn(),
      getInitProgressCallback: jest.fn(() => initCallback),
      setInitProgressCallback: jest.fn(),
      setLogitProcessorRegistry: jest.fn(),
      setAppConfig: jest.fn(),
    })),
  };
});

const originalNavigator = (globalThis as any).navigator;
const originalSelf = (globalThis as any).self;
const originalPostMessage = (globalThis as any).postMessage;

function setupWorkerScope() {
  (globalThis as any).self = {
    addEventListener: jest.fn(),
  };
  (globalThis as any).postMessage = jest.fn();
}

function setupNavigator(options?: {
  controller?: any;
  readyRegistration?: Promise<ServiceWorkerRegistration>;
}) {
  const controller = options?.controller ?? { postMessage: jest.fn() };
  const readyRegistration =
    options?.readyRegistration ??
    Promise.resolve({
      active: controller as ServiceWorker,
    } as unknown as ServiceWorkerRegistration);
  const container: any = {
    controller,
    ready: readyRegistration,
    onmessage: undefined as any,
  };
  (globalThis as any).navigator = {
    serviceWorker: container,
  };
  return { container, controller };
}

function createHandler() {
  setupWorkerScope();
  const handler = new ServiceWorkerMLCEngineHandler();
  const handleTaskMock = jest.fn(async (_uuid: string, task: any) => task());
  (handler as any).handleTask = handleTaskMock;
  (handler as any).engine = {
    reload: reloadMock,
    getInitProgressCallback: jest.fn(() => initCallback),
  };
  reloadMock.mockClear();
  initCallback.mockClear();
  return { handler, handleTaskMock };
}

afterEach(() => {
  reloadMock.mockClear();
  initCallback.mockClear();
  if (originalNavigator === undefined) {
    delete (globalThis as any).navigator;
  } else {
    (globalThis as any).navigator = originalNavigator;
  }
  if (originalSelf === undefined) {
    delete (globalThis as any).self;
  } else {
    (globalThis as any).self = originalSelf;
  }
  if (originalPostMessage === undefined) {
    delete (globalThis as any).postMessage;
  } else {
    (globalThis as any).postMessage = originalPostMessage;
  }
  jest.useRealTimers();
  jest.clearAllTimers();
});

test("ServiceWorker handler responds to keepAlive message", () => {
  const { handler } = createHandler();
  const client = { postMessage: jest.fn() };
  (handler as any).clientRegistry = new Map([["keep", client]]);
  const onComplete = jest.fn();
  handler.onmessage(
    { data: { kind: "keepAlive", uuid: "keep" } } as ServiceWorkerHandlerEvent,
    onComplete,
  );
  expect(client.postMessage).toHaveBeenCalledWith({
    kind: "heartbeat",
    uuid: "keep",
  });
  expect(onComplete).toHaveBeenCalledWith({ kind: "heartbeat", uuid: "keep" });
});

test("reload with the same model skips engine reload", async () => {
  const { handler, handleTaskMock } = createHandler();
  handler.modelId = ["demo"];
  handler.chatOpts = [];
  handler.onmessage({
    data: {
      kind: "reload",
      uuid: "reload-same",
      content: { modelId: ["demo"], chatOpts: [] },
    },
  } as ServiceWorkerHandlerEvent);
  await handleTaskMock.mock.results[0].value;
  expect(reloadMock).not.toHaveBeenCalled();
  expect(initCallback).toHaveBeenCalledWith(
    expect.objectContaining({ progress: 1 }),
  );
});

test("reload with new parameters calls engine reload", async () => {
  const { handler, handleTaskMock } = createHandler();
  handler.modelId = ["demo"];
  handler.chatOpts = [];
  handler.onmessage({
    data: {
      kind: "reload",
      uuid: "reload-new",
      content: { modelId: ["fresh"], chatOpts: [] },
    },
  } as ServiceWorkerHandlerEvent);
  await handleTaskMock.mock.results[0].value;
  expect(reloadMock).toHaveBeenCalledWith(["fresh"], []);
});

test("ServiceWorker client forwards onmessage handlers to navigator", () => {
  const { container } = setupNavigator();
  const client = new ServiceWorker();
  const handler = jest.fn();
  client.onmessage = handler;
  expect(container.onmessage).toBe(handler);
});

test("ServiceWorker postMessage routes through controller", () => {
  const { controller } = setupNavigator();
  const client = new ServiceWorker();
  const message = { kind: "reload", uuid: "client-msg" } as any;
  client.postMessage(message);
  expect(controller.postMessage).toHaveBeenCalledWith(message);
});

test("ServiceWorker postMessage throws if controller missing", () => {
  const { container } = setupNavigator(undefined);
  container.controller = undefined as any;
  const client = new ServiceWorker();
  expect(() =>
    client.postMessage({ kind: "reload", uuid: "client-msg" } as any),
  ).toThrow("There is no active service worker");
});

test("ServiceWorkerMLCEngine heartbeats reset missed counter", () => {
  jest.useFakeTimers();
  const { container } = setupNavigator();
  const engine = new ServiceWorkerMLCEngine(undefined, 200);
  expect(engine.missedHeartbeat).toBe(0);
  jest.advanceTimersByTime(200);
  expect((container.controller as any).postMessage).toHaveBeenCalledWith(
    expect.objectContaining({ kind: "keepAlive" }),
  );
  expect(engine.missedHeartbeat).toBe(1);
  container.onmessage?.({ data: { kind: "heartbeat" } } as MessageEvent<any>);
  expect(engine.missedHeartbeat).toBe(0);
});

test("CreateServiceWorkerMLCEngine waits for ready registration and reloads", async () => {
  jest.useFakeTimers();
  setupNavigator();
  const reloadSpy = jest
    .spyOn(ServiceWorkerMLCEngine.prototype, "reload")
    .mockResolvedValue(undefined);
  const engine = await CreateServiceWorkerMLCEngine(["m1", "m2"]);
  expect(reloadSpy).toHaveBeenCalledWith(["m1", "m2"], undefined);
  expect(engine).toBeInstanceOf(ServiceWorkerMLCEngine);
  reloadSpy.mockRestore();
});
