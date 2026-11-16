import * as tvmjs from "@mlc-ai/web-runtime";
import type { ModelRecord } from "./config";
import {
  FeatureSupportError,
  ShaderF16SupportError,
  WebGPUNotAvailableError,
} from "./error";

/** Select a device that satisfies a model's declared buffer requirement. */
export async function detectGPUDeviceForModel(
  record: Pick<ModelRecord, "buffer_size_required_bytes" | "required_features">,
  signal?: AbortSignal,
): Promise<tvmjs.GPUDeviceDetectOutput> {
  const checkAborted = () => {
    if (signal?.aborted)
      throw new DOMException("Reload was aborted.", "AbortError");
  };
  checkAborted();
  const declared = record.buffer_size_required_bytes;
  const required =
    declared === undefined ? undefined : Math.ceil(declared / 4) * 4;
  if (
    declared !== undefined &&
    (!Number.isSafeInteger(declared) ||
      declared <= 0 ||
      !Number.isSafeInteger(required))
  ) {
    throw new RangeError(
      "buffer_size_required_bytes must be a positive safe integer whose four-byte alignment is safe",
    );
  }
  const detected = await tvmjs.detectGPUDevice();
  if (detected === undefined) {
    checkAborted();
    throw new WebGPUNotAvailableError();
  }
  let device = detected.device;
  let adapter = detected.adapter;
  let adapterInfo = detected.adapterInfo;
  const checkFeatures = () => {
    for (const feature of record.required_features ?? []) {
      if (!device.features.has(feature as GPUFeatureName)) {
        if (feature === "shader-f16") {
          throw new ShaderF16SupportError();
        }
        throw new FeatureSupportError(feature);
      }
    }
  };
  try {
    checkAborted();
    checkFeatures();
    if (
      required !== undefined &&
      (required > device.limits.maxBufferSize ||
        required > device.limits.maxStorageBufferBindingSize)
    ) {
      // A GPUAdapter can be consumed by its first requestDevice call.
      const freshAdapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      checkAborted();
      if (freshAdapter === null) {
        throw new WebGPUNotAvailableError();
      }
      adapter = freshAdapter;
      if (freshAdapter.info) {
        adapterInfo = freshAdapter.info;
      } else {
        const legacy = freshAdapter as GPUAdapter & {
          requestAdapterInfo(): Promise<GPUAdapterInfo>;
        };
        adapterInfo = await legacy.requestAdapterInfo();
        checkAborted();
      }
      const maxBufferSize = Math.max(required, device.limits.maxBufferSize);
      const maxStorageBufferBindingSize = Math.max(
        required,
        device.limits.maxStorageBufferBindingSize,
      );
      if (
        maxBufferSize > adapter.limits.maxBufferSize ||
        maxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize
      ) {
        throw new Error(
          `Model requires WebGPU buffer capacity ${required} bytes, but the adapter ` +
            `supports maxBufferSize=${adapter.limits.maxBufferSize} and ` +
            `maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize}`,
        );
      }
      const replacement = await adapter.requestDevice({
        requiredFeatures: Array.from(device.features) as GPUFeatureName[],
        requiredLimits: {
          maxBufferSize,
          maxStorageBufferBindingSize,
          // Preserve the additional limits requested by web-runtime's detector.
          maxComputeWorkgroupStorageSize:
            device.limits.maxComputeWorkgroupStorageSize,
          maxStorageBuffersPerShaderStage:
            device.limits.maxStorageBuffersPerShaderStage,
          maxComputeInvocationsPerWorkgroup:
            device.limits.maxComputeInvocationsPerWorkgroup,
        },
      });
      const provisional = device;
      device = replacement;
      provisional.destroy();
      checkAborted();
      if (
        device.limits.maxBufferSize < maxBufferSize ||
        device.limits.maxStorageBufferBindingSize < maxStorageBufferBindingSize
      ) {
        throw new Error(
          "WebGPU device does not satisfy the requested buffer limits",
        );
      }
      checkFeatures();
    }
    return { adapter, adapterInfo, device };
  } catch (error) {
    try {
      device.destroy();
    } catch {
      // Keep the original device-selection error.
    }
    throw error;
  }
}
