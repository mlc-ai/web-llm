import {
  ensureNonblockingWebGPUCommandSubmission,
  ensureSampledTokenReadbackRingRuntime,
  NONBLOCKING_WEBGPU_SUBMISSION_RUNTIME_HINT,
  submitPendingWebGPUCommands,
  submitSampledTokenReadbackBatch,
  supportsNonblockingWebGPUCommandSubmission,
  supportsSampledTokenReadbackRingRuntime,
  SAMPLED_TOKEN_READBACK_BATCH_RUNTIME_HINT,
  SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT,
} from "../src/runtime_readback";
import { describe, expect, jest, test } from "@jest/globals";

// These guard the runtime feature detection: an older @mlc-ai/web-runtime
// must produce the named error instead of a TypeError from a missing method.
describe("runtime_readback compatibility helpers", () => {
  test("ring support is detected by the device method and reported when missing", () => {
    const baselineOnlyDevice = {} as any;
    expect(supportsSampledTokenReadbackRingRuntime(baselineOnlyDevice)).toBe(
      false,
    );
    expect(() =>
      ensureSampledTokenReadbackRingRuntime(baselineOnlyDevice),
    ).toThrow(SAMPLED_TOKEN_READBACK_RING_RUNTIME_HINT);
    const ringCapableDevice = {
      createSampledTokenReadbackRing: () => null,
    } as any;
    expect(supportsSampledTokenReadbackRingRuntime(ringCapableDevice)).toBe(
      true,
    );
  });

  test("nonblocking submission is detected by the device method", () => {
    expect(supportsNonblockingWebGPUCommandSubmission({} as any)).toBe(false);
    expect(() => ensureNonblockingWebGPUCommandSubmission({} as any)).toThrow(
      NONBLOCKING_WEBGPU_SUBMISSION_RUNTIME_HINT,
    );
    const submitPendingCommands = jest.fn(() => true);
    expect(submitPendingWebGPUCommands({ submitPendingCommands } as any)).toBe(
      true,
    );
    expect(submitPendingCommands).toHaveBeenCalledTimes(1);
  });

  test("a batch uses submit for one token and submitTokens for several", () => {
    const token = {} as any;
    const ring = {
      submit: jest.fn<(...args: unknown[]) => number>(() => 7),
      submitTokens: jest.fn<(...args: unknown[]) => number>(() => 8),
    };
    expect(submitSampledTokenReadbackBatch(ring as any, [token])).toBe(7);
    expect(ring.submit).toHaveBeenCalledWith(token, 1, 0);
    expect(submitSampledTokenReadbackBatch(ring as any, [token, token])).toBe(
      8,
    );
    expect(() =>
      submitSampledTokenReadbackBatch({ submit: jest.fn() } as any, [
        token,
        token,
      ]),
    ).toThrow(SAMPLED_TOKEN_READBACK_BATCH_RUNTIME_HINT);
  });
});
