import * as tvmjs from "@mlc-ai/web-runtime";

export const SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT =
  'sample_readback_mode="ring_vector" requires a web runtime with sampled-token readback ring support.';

export const SAMPLED_TOKEN_READBACK_BATCH_RUNTIME_HINT =
  "sample_readback_batch_size > 1 requires an @mlc-ai/web-runtime build whose " +
  "sampled-token readback ring exposes submitTokens().";

export const NONBLOCKING_WEBGPU_SUBMISSION_RUNTIME_HINT =
  "sample_readback_batch_size > 1 requires " +
  "an @mlc-ai/web-runtime build whose WebGPU device exposes submitPendingCommands().";

/** One drained ring batch: `tokenCount` valid int32 token ids in `tokens`. */
export interface SampledTokenReadbackBatch {
  batchSeq: number;
  tokenCount: number;
  tokens: Int32Array;
}

export interface SampledTokenReadbackRingOptions {
  slotCount?: number;
  maxTokensPerBatch: number;
}

/** The ring handle the TVM web runtime returns from createSampledTokenReadbackRing. */
export interface SampledTokenReadbackRing {
  submit(
    from: tvmjs.Tensor,
    tokenCount: number,
    fromTokenOffset?: number,
    into?: Int32Array,
  ): number;
  submitFromPointer(
    fromPtr: number,
    tokenCount: number,
    fromTokenOffset?: number,
    into?: Int32Array,
  ): number;
  pollReady(): Array<SampledTokenReadbackBatch>;
  waitReady(): Promise<Array<SampledTokenReadbackBatch>>;
  dispose(): void;
}

type DLDeviceWithSampledTokenReadbackRing = tvmjs.DLDevice & {
  createSampledTokenReadbackRing: (
    options: SampledTokenReadbackRingOptions,
  ) => SampledTokenReadbackRing;
};

type SampledTokenReadbackRingWithBatchSubmit = SampledTokenReadbackRing & {
  submitTokens: (tokens: Array<tvmjs.Tensor>) => number;
};

type DLDeviceWithNonblockingWebGPUSubmission = tvmjs.DLDevice & {
  submitPendingCommands: () => boolean;
};

export function supportsSampledTokenReadbackRingRuntime(
  device: tvmjs.DLDevice,
): device is DLDeviceWithSampledTokenReadbackRing {
  return (
    typeof (device as Partial<DLDeviceWithSampledTokenReadbackRing>)
      .createSampledTokenReadbackRing === "function"
  );
}

export function ensureSampledTokenReadbackRingRuntime(
  device: tvmjs.DLDevice,
): asserts device is DLDeviceWithSampledTokenReadbackRing {
  if (!supportsSampledTokenReadbackRingRuntime(device)) {
    throw new Error(SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT);
  }
}

export function createSampledTokenReadbackRing(
  device: tvmjs.DLDevice,
  options: SampledTokenReadbackRingOptions,
): SampledTokenReadbackRing {
  ensureSampledTokenReadbackRingRuntime(device);
  return device.createSampledTokenReadbackRing(options);
}

export function supportsNonblockingWebGPUCommandSubmission(
  device: tvmjs.DLDevice,
): device is DLDeviceWithNonblockingWebGPUSubmission {
  return (
    typeof (device as Partial<DLDeviceWithNonblockingWebGPUSubmission>)
      .submitPendingCommands === "function"
  );
}

export function ensureNonblockingWebGPUCommandSubmission(
  device: tvmjs.DLDevice,
): asserts device is DLDeviceWithNonblockingWebGPUSubmission {
  if (!supportsNonblockingWebGPUCommandSubmission(device)) {
    throw new Error(NONBLOCKING_WEBGPU_SUBMISSION_RUNTIME_HINT);
  }
}

export function submitPendingWebGPUCommands(device: tvmjs.DLDevice): boolean {
  ensureNonblockingWebGPUCommandSubmission(device);
  return device.submitPendingCommands();
}

export function supportsSampledTokenReadbackBatchSubmit(
  ring: SampledTokenReadbackRing,
): boolean {
  return (
    typeof (ring as Partial<SampledTokenReadbackRingWithBatchSubmit>)
      .submitTokens === "function"
  );
}

export function submitSampledTokenReadbackBatch(
  ring: SampledTokenReadbackRing,
  tokens: Array<tvmjs.Tensor>,
): number {
  if (tokens.length === 0) {
    throw new Error("Cannot submit an empty sampled-token readback batch.");
  }
  if (tokens.length === 1) {
    return ring.submit(tokens[0], 1, 0);
  }
  if (!supportsSampledTokenReadbackBatchSubmit(ring)) {
    throw new Error(SAMPLED_TOKEN_READBACK_BATCH_RUNTIME_HINT);
  }
  return (ring as SampledTokenReadbackRingWithBatchSubmit).submitTokens(tokens);
}

/** Read raw 32-bit words, including the bits of floating-point tensors. */
export function submitTensorReadbackWords(
  ring: SampledTokenReadbackRing,
  from: tvmjs.Tensor,
  wordCount: number,
  into?: Int32Array,
): number {
  const { byteOffset } = from as unknown as { readonly byteOffset: number };
  return ring.submitFromPointer(
    from.getDataPtr(),
    wordCount,
    byteOffset / 4,
    into,
  );
}
