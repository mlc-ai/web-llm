import {
  CreateServiceWorkerMLCEngine,
  ServiceWorkerMLCEngine,
  ServiceWorkerMLCEngineHandler,
} from "../src/extension_service_worker";
import {
  jest,
  test,
  expect,
  describe,
  beforeEach,
  afterEach,
} from "@jest/globals";

jest.mock("@mlc-ai/web-runtime", () => ({
  detectGPUDevice: jest.fn(async () => ({
    adapterInfo: { description: "MockGPU", vendor: "MockVendor" },
    device: { features: new Set() },
  })),
}));

const reloadMock = jest.fn();
const initCallback = jest.fn();

jest.mock("../src/engine", () => {
  return {
    MLCEngine: jest.fn(() => ({
      reload: reloadMock,
      getInitProgressCallback: jest.fn(() => initCallback),
      setInitProgressCallback: jest.fn(),
    })),
  };
});

type MockPort = chrome.runtime.Port & {
  triggerDisconnect: () => void;
  emitMessage: (msg: any) => void;
};

function createPort(): MockPort {
  const disconnectListeners: Array<() => void> = [];
  const messageListeners: Array<(msg: any) => void> = [];
  return {
    postMessage: jest.fn(),
    onDisconnect: {
      addListener: (cb: () => void) => disconnectListeners.push(cb),
    },
    onMessage: {
      addListener: (cb: (msg: any) => void) => messageListeners.push(cb),
    },
    triggerDisconnect: () => disconnectListeners.forEach((cb) => cb()),
    emitMessage: (msg: any) => messageListeners.forEach((cb) => cb(msg)),
  } as unknown as MockPort;
}

function createHandler() {
  const handler = new ServiceWorkerMLCEngineHandler(createPort());
  (handler as any).handleTask = jest.fn(async (_uuid: string, task: any) =>
    task(),
  );
  (handler as any).engine = {
    reload: reloadMock,
    getInitProgressCallback: jest.fn(() => initCallback),
  };
  reloadMock.mockClear();
  initCallback.mockClear();
  return handler;
}

test("reload message with same model skips loading and triggers init callback", async () => {
  const handler = createHandler();
  handler.modelId = ["demo"];
  handler.chatOpts = [];
  await handler.onmessage({
    type: "message",
    kind: "reload",
    uuid: "task",
    content: { modelId: ["demo"], chatOpts: [] },
  } as any);
  expect(reloadMock).not.toHaveBeenCalled();
  expect(initCallback).toHaveBeenCalled();
});

test("reload with new model calls engine reload", async () => {
  const handler = createHandler();
  handler.modelId = ["demo"];
  handler.chatOpts = [];
  await handler.onmessage({
    kind: "reload",
    uuid: "task",
    content: { modelId: ["new"], chatOpts: [] },
  } as any);
  expect(reloadMock).toHaveBeenCalledWith(["new"], []);
});

function mockChromeRuntime(port: MockPort = createPort()) {
  const connect = jest.fn<(...args: any[]) => MockPort>(() => port);
  (globalThis as any).chrome = {
    runtime: {
      connect,
    },
  };
  return { port, connect };
}

describe("ServiceWorkerMLCEngine integration", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllTimers();
    delete (globalThis as any).chrome;
  });

  test("keepAlive pings and onDisconnect callback fires", () => {
    const { port, connect } = mockChromeRuntime();
    const onDisconnect = jest.fn();
    const engine = new ServiceWorkerMLCEngine({ onDisconnect }, 500);
    expect(connect).toHaveBeenCalledWith({ name: "web_llm_service_worker" });
    jest.advanceTimersByTime(500);
    expect(port.postMessage).toHaveBeenCalledWith({ kind: "keepAlive" });
    port.triggerDisconnect();
    expect(onDisconnect).toHaveBeenCalled();
    expect(engine).toBeTruthy();
  });

  test("CreateServiceWorkerMLCEngine reloads requested model", async () => {
    const { connect } = mockChromeRuntime();
    const reloadSpy = jest
      .spyOn(ServiceWorkerMLCEngine.prototype, "reload")
      .mockResolvedValue(undefined);
    const engine = await CreateServiceWorkerMLCEngine("demo-model", {
      extensionId: "abc",
    });
    expect(connect).toHaveBeenCalledWith("abc", {
      name: "web_llm_service_worker",
    });
    expect(reloadSpy).toHaveBeenCalledWith("demo-model", undefined);
    reloadSpy.mockRestore();
    expect(engine).toBeInstanceOf(ServiceWorkerMLCEngine);
  });
});
