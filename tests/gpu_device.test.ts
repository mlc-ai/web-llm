import { afterAll, beforeEach, expect, jest, test } from "@jest/globals";
import * as tvmjs from "@mlc-ai/web-runtime";
import { detectGPUDeviceForModel } from "../src/gpu_device";

jest.mock("@mlc-ai/web-runtime", () => ({ detectGPUDevice: jest.fn() }));
const detect = jest.mocked(tvmjs.detectGPUDevice);
const GiB = 2 ** 30;
const requestAdapter = jest.fn<GPU["requestAdapter"]>();
const originalNavigator = Object.getOwnPropertyDescriptor(
  globalThis,
  "navigator",
);
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { gpu: { requestAdapter } },
});
afterAll(() => {
  if (originalNavigator)
    Object.defineProperty(globalThis, "navigator", originalNavigator);
  else Reflect.deleteProperty(globalThis, "navigator");
});
function device(buffer = GiB, binding = GiB, features = ["shader-f16"]) {
  return {
    limits: {
      maxBufferSize: buffer,
      maxStorageBufferBindingSize: binding,
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeInvocationsPerWorkgroup: 1024,
    },
    features: new Set(features),
    destroy: jest.fn(),
  } as unknown as GPUDevice;
}
function fixture() {
  const original = device();
  const replacement = device(2 * GiB, 2 * GiB);
  const requestDevice = jest
    .fn<GPUAdapter["requestDevice"]>()
    .mockResolvedValue(replacement);
  const adapter = {
    limits: { maxBufferSize: 4 * GiB, maxStorageBufferBindingSize: 4 * GiB },
    requestDevice,
    info: {} as GPUAdapterInfo,
  } as unknown as GPUAdapter;
  const detected = {
    device: original,
    adapter,
    adapterInfo: {} as GPUAdapterInfo,
  };
  requestAdapter.mockResolvedValue(adapter);
  detected.adapter = {
    ...adapter,
    requestDevice: jest
      .fn<GPUAdapter["requestDevice"]>()
      .mockRejectedValue(new Error("adapter consumed")),
  } as GPUAdapter;
  detect.mockResolvedValue(detected);
  return { original, replacement, requestDevice, detected, adapter };
}
beforeEach(() => {
  jest.clearAllMocks();
});

test.each([undefined, 256 * 1024 * 1024, GiB])(
  "preserves adequate default device (%s)",
  async (size) => {
    const f = fixture();
    const actual = await detectGPUDeviceForModel({
      buffer_size_required_bytes: size,
    });
    expect(actual.device).toBe(f.original);
    expect(f.requestDevice).not.toHaveBeenCalled();
    expect(f.original.destroy).not.toHaveBeenCalled();
  },
);

test.each([0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])(
  "rejects invalid requirement %s before creating a device",
  async (size) => {
    await expect(
      detectGPUDeviceForModel({ buffer_size_required_bytes: size }),
    ).rejects.toThrow("positive safe integer");
    expect(detect).not.toHaveBeenCalled();
  },
);

test("requests exact effective capacities and retains runtime features and limits", async () => {
  const f = fixture();
  const required = 1271398400;
  const result = await detectGPUDeviceForModel({
    buffer_size_required_bytes: required,
    required_features: ["shader-f16"],
  });
  expect(result.device).toBe(f.replacement);
  expect(f.requestDevice).toHaveBeenCalledWith({
    requiredFeatures: ["shader-f16"],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeInvocationsPerWorkgroup: 1024,
      maxBufferSize: required,
      maxStorageBufferBindingSize: required,
    },
  });
  expect(f.detected.adapter.requestDevice).not.toHaveBeenCalled();
  expect(requestAdapter).toHaveBeenCalledWith({
    powerPreference: "high-performance",
  });
  expect(f.original.destroy).toHaveBeenCalledTimes(1);
  expect(f.replacement.destroy).not.toHaveBeenCalled();
});

test("preserves a larger existing maxBufferSize when only binding capacity is insufficient", async () => {
  const f = fixture();
  const original = device(2 * GiB, GiB);
  detect.mockResolvedValue({ ...f.detected, device: original });
  await detectGPUDeviceForModel({ buffer_size_required_bytes: 1271398400 });
  expect(f.requestDevice.mock.calls[0][0]?.requiredLimits).toMatchObject({
    maxBufferSize: 2 * GiB,
    maxStorageBufferBindingSize: 1271398400,
  });
});

test.each(["maxBufferSize", "maxStorageBufferBindingSize"] as const)(
  "rejects adapter %s limit and destroys provisional device",
  async (limit) => {
    const f = fixture();
    Object.assign(f.detected.adapter.limits, { [limit]: GiB });
    await expect(
      detectGPUDeviceForModel({ buffer_size_required_bytes: 2 * GiB }),
    ).rejects.toThrow("adapter");
    expect(f.requestDevice).not.toHaveBeenCalled();
    expect(f.original.destroy).toHaveBeenCalledTimes(1);
  },
);

test("cleans up after replacement request rejection", async () => {
  const f = fixture();
  const failure = new Error("request failed");
  f.requestDevice.mockRejectedValue(failure);
  await expect(
    detectGPUDeviceForModel({ buffer_size_required_bytes: 2 * GiB }),
  ).rejects.toBe(failure);
  expect(f.original.destroy).toHaveBeenCalledTimes(1);
});

test("rejects insufficient effective replacement limits and cleans up both devices", async () => {
  const f = fixture();
  const replacement = device(2 * GiB, GiB);
  f.requestDevice.mockResolvedValue(replacement);
  await expect(
    detectGPUDeviceForModel({ buffer_size_required_bytes: 2 * GiB }),
  ).rejects.toThrow("requested buffer limits");
  expect(f.original.destroy).toHaveBeenCalledTimes(1);
  expect(replacement.destroy).toHaveBeenCalledTimes(1);
});

test.each(["shader-f16", "subgroups"])(
  "cleans up when required feature %s is unavailable",
  async (feature) => {
    const f = fixture();
    const original = device(GiB, GiB, []);
    detect.mockResolvedValue({ ...f.detected, device: original });
    await expect(
      detectGPUDeviceForModel({ required_features: [feature] }),
    ).rejects.toThrow();
    expect(original.destroy).toHaveBeenCalledTimes(1);
  },
);

test("cleans up a replacement missing a required feature", async () => {
  const f = fixture();
  const replacement = device(2 * GiB, 2 * GiB, []);
  f.requestDevice.mockResolvedValue(replacement);
  await expect(
    detectGPUDeviceForModel({
      buffer_size_required_bytes: 2 * GiB,
      required_features: ["shader-f16"],
    }),
  ).rejects.toThrow();
  expect(f.original.destroy).toHaveBeenCalledTimes(1);
  expect(replacement.destroy).toHaveBeenCalledTimes(1);
});

test("reports unavailable WebGPU", async () => {
  detect.mockResolvedValue(undefined);
  await expect(detectGPUDeviceForModel({})).rejects.toThrow("WebGPU");
});

test("rounds required capacities up to the runtime's four-byte alignment", async () => {
  const f = fixture();
  await detectGPUDeviceForModel({ buffer_size_required_bytes: GiB + 1 });
  expect(f.requestDevice.mock.calls[0][0]?.requiredLimits).toMatchObject({
    maxBufferSize: GiB + 4,
    maxStorageBufferBindingSize: GiB + 4,
  });
});

test("rejects alignment beyond the safe integer range", async () => {
  await expect(
    detectGPUDeviceForModel({
      buffer_size_required_bytes: Number.MAX_SAFE_INTEGER,
    }),
  ).rejects.toThrow("alignment");
  expect(detect).not.toHaveBeenCalled();
});

test("cleans up when no fresh adapter is available", async () => {
  const f = fixture();
  requestAdapter.mockResolvedValue(null);
  await expect(
    detectGPUDeviceForModel({ buffer_size_required_bytes: 2 * GiB }),
  ).rejects.toThrow("WebGPU");
  expect(f.original.destroy).toHaveBeenCalledTimes(1);
});

test("preserves the dev2 legacy adapter info fallback", async () => {
  const f = fixture();
  const info = { description: "legacy" } as GPUAdapterInfo;
  const legacyInfo = jest
    .fn<() => Promise<GPUAdapterInfo>>()
    .mockResolvedValue(info);
  Object.assign(f.adapter, { info: undefined, requestAdapterInfo: legacyInfo });
  const actual = await detectGPUDeviceForModel({
    buffer_size_required_bytes: 2 * GiB,
  });
  expect(actual.adapterInfo).toBe(info);
  expect(legacyInfo).toHaveBeenCalledTimes(1);
});

test("abort during legacy adapter info prevents a replacement device and cleans up", async () => {
  const f = fixture();
  const controller = new AbortController();
  Object.assign(f.adapter, {
    info: undefined,
    requestAdapterInfo: async () => {
      controller.abort();
      return {} as GPUAdapterInfo;
    },
  });
  await expect(
    detectGPUDeviceForModel(
      { buffer_size_required_bytes: 2 * GiB },
      controller.signal,
    ),
  ).rejects.toMatchObject({ name: "AbortError" });
  expect(f.requestDevice).not.toHaveBeenCalled();
  expect(f.original.destroy).toHaveBeenCalledTimes(1);
});

test("an already aborted selection does not create a device", async () => {
  const controller = new AbortController();
  controller.abort();
  await expect(
    detectGPUDeviceForModel({}, controller.signal),
  ).rejects.toMatchObject({ name: "AbortError" });
  expect(detect).not.toHaveBeenCalled();
});
