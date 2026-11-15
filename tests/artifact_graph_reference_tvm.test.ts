import { createHash, webcrypto } from "crypto";
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createReferenceConditionerTVMFactory } from "../src/artifact_graph_reference_tvm";
import {
  computeGraphInterfaceId,
  ModelPackageManifestV2,
  CompiledProgramArtifactV2,
} from "../src/artifact_manifest_v2";
import { GraphInputEvent } from "../src/artifact_graph_executor";
beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  }),
);
async function fixture() {
  const schema =
    "sha256:" +
    createHash("sha256")
      .update('[{"dtype":"float32","name":"weight","shape":[1]}]')
      .digest("hex");
  const input = {
    semantic: "audio.reference_mel.v1",
    dtype: "float32" as const,
    shape: ["frames", 2],
    delivery: "value" as const,
  };
  const output = {
    semantic: "audio.reference_embedding.v1",
    dtype: "float32" as const,
    shape: [3],
    delivery: "value" as const,
  };
  const spec = {
    kind: "encoder" as const,
    contract: "qwen2_5_omni.reference_conditioner.v1",
    inputs: { mel: input },
    outputs: { embedding: output },
    weights: { manifest: "tensor-cache.json", parameter_schema_id: schema },
    state: { kind: "none" as const, scope: "request" as const },
  };
  const model: ModelPackageManifestV2 = {
    schema: "mlc.model-package",
    schema_version: 2,
    interface_id: "sha256:" + "0".repeat(64),
    components: { audio: spec },
    tasks: {
      decode: {
        inputs: {
          mel: {
            spec: input,
            targets: [{ component: "audio", port: "mel" }],
          },
        },
        outputs: {
          embedding: {
            spec: output,
            source: { component: "audio", port: "embedding" },
            optional: false,
          },
        },
        edges: [],
      },
    },
  };
  model.interface_id = await computeGraphInterfaceId(model);
  const compiled: CompiledProgramArtifactV2 = {
    schema: "mlc.compiled-program",
    schema_version: 2,
    interface_id: model.interface_id,
    components: {
      audio: {
        spec: model.components.audio,
        module: "audio.wasm",
        exports: { run: "forward", forward: "forward" },
        resources: {
          required_features: [],
          max_storage_buffer_binding_size: 0,
          estimated_device_memory_bytes: 0,
        },
      },
    },
  };
  return { model, compiled };
}
class Tensor {
  data: Float32Array;
  disposed = false;
  constructor(
    public shape: number[],
    public dtype = "float32",
  ) {
    this.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  }
  copyFrom(data: Float32Array | Int32Array | Tensor) {
    this.data.set(data instanceof Tensor ? data.data : data);
    return this;
  }
  toArray() {
    return this.data.slice();
  }
  dispose() {
    this.disposed = true;
  }
}
async function setup() {
  const { model, compiled } = await fixture();
  const metadata = {
    artifact: compiled,
    params: [{ name: "weight", shape: [1], dtype: "float32" }],
    reference_conditioner: {
      mel_dim: 2,
      embedding_dim: 3,
      min_mel_frames: 5,
      max_mel_frames: 32,
    },
  };
  const calls: { name: string; shape: number[]; data: number[] }[] = [];
  const packed = { dispose: jest.fn() };
  const tensors: Tensor[] = [];
  const scopes: { dispose(): void }[][] = [];
  const track = <T extends { dispose(): void }>(v: T): T => {
    scopes[scopes.length - 1].push(v);
    return v;
  };
  const funcs = {
    _metadata: () => ({ toString: () => JSON.stringify(metadata) }),
    forward: (x: Tensor, p: unknown) => {
      expect(p).toBe(packed);
      calls.push({ name: "forward", shape: x.shape, data: Array.from(x.data) });
      const out = track(new Tensor([3]));
      out.data.fill(2.5);
      return out;
    },
  };
  const vm = {
    dispose: jest.fn(),
    getFunction: (name: string) =>
      track(
        Object.assign(funcs[name as keyof typeof funcs], {
          dispose: jest.fn(),
        }),
      ),
  };
  const device = { sync: jest.fn(async () => {}) };
  const runtime = {
    beginScope: () => scopes.push([]),
    endScope: () =>
      scopes
        .pop()!
        .reverse()
        .forEach((v) => v.dispose()),
    detachFromCurrentScope: <T extends { dispose(): void }>(v: T) => {
      const s = scopes[scopes.length - 1];
      s.splice(s.indexOf(v), 1);
      return v;
    },
    createVirtualMachine: () => track(vm),
    tensorCacheGet: () => track(new Tensor([1])),
    getParamsFromCacheByName: jest.fn(() => track(packed)),
    empty: (shape: number[], dtype: string) => {
      const tensor = track(new Tensor(shape, dtype));
      tensors.push(tensor);
      return tensor;
    },
    cpu: () => ({}),
  };
  const options = {
    runtime: runtime as unknown as tvmjs.Instance,
    device: device as unknown as tvmjs.DLDevice,
    model,
    compiled,
    componentId: "audio",
    maxFrames: 32,
    capabilities: {
      features: new Set<string>(),
      maxStorageBufferBindingSize: 1024,
    },
  };
  const factory = await createReferenceConditionerTVMFactory(options);
  return {
    factory,
    tensors,
    options,
    compiled,
    metadata,
    calls,
    device,
    vm,
    packed,
    runtime,
    funcs,
  };
}

async function* input(frames = 5, value = 0): AsyncIterable<GraphInputEvent> {
  yield {
    kind: "data",
    port: "mel",
    tensor: {
      shape: [frames, 2],
      data: new Float32Array(2 * frames).fill(value),
    },
  };
  yield { kind: "end", port: "mel" };
}
async function execute(
  s: Awaited<ReturnType<typeof setup>>,
  events = input(),
  controller = new AbortController(),
) {
  const backend = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    controller.signal,
  );
  try {
    const out = [];
    for await (const event of backend.run(events, controller.signal))
      out.push(event);
    return out;
  } finally {
    await backend.dispose();
  }
}
test("invokes packed forward with exact clip and returns an unclamped nonzero null-mel embedding", async () => {
  const s = await setup(),
    out = await execute(s);
  expect(s.calls).toEqual([
    { name: "forward", shape: [5, 2], data: Array(10).fill(0) },
  ]);
  expect(out[0].tensor.shape).toEqual([3]);
  expect(Array.from(out[0].tensor.data)).toEqual(Array(3).fill(2.5));
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
  expect(s.packed.dispose).toHaveBeenCalledTimes(1);
  await execute(s); // A disposed backend releases exclusive runtime ownership.
});
test.each([
  [0, 0],
  [4, 0],
  [33, 0],
  [5, NaN],
  [5, Infinity],
])("rejects invalid clip %s/%s before forward", async (frames, value) => {
  const s = await setup();
  await expect(execute(s, input(frames, value))).rejects.toThrow(
    "Invalid reference-mel",
  );
  expect(s.calls).toHaveLength(0);
});
test("rejects missing end event", async () => {
  const s = await setup();
  async function* events() {
    yield (await input()[Symbol.asyncIterator]().next()).value!;
  }
  await expect(execute(s, events())).rejects.toThrow("Missing reference-mel");
});
test("checks cancellation after synchronization and releases output", async () => {
  const s = await setup(),
    abort = new AbortController();
  s.device.sync.mockImplementation(async () => {
    abort.abort();
  });
  await expect(execute(s, input(), abort)).rejects.toThrow("aborted");
  expect(s.vm.dispose).toHaveBeenCalled();
});
test("checks cache shape and compiled frame metadata", async () => {
  const s = await setup();
  s.metadata.reference_conditioner.max_mel_frames = 1;
  await expect(execute(s)).rejects.toThrow("metadata");
  expect(s.calls).toHaveLength(0);
});
test("rejects incompatible contract semantics", async () => {
  const s = await setup();
  s.options.model.components.audio.inputs.mel.semantic = "audio.mel.v1";
  s.options.model.interface_id = await computeGraphInterfaceId(s.options.model);
  s.compiled.interface_id = s.options.model.interface_id;
  await expect(createReferenceConditionerTVMFactory(s.options)).rejects.toThrow(
    "ABI",
  );
});

test("scope cleanup failure releases detached CPU output and runtime ownership", async () => {
  const s = await setup(),
    end = s.runtime.endScope;
  s.runtime.endScope = () => {
    end();
    if (s.calls.length) throw new Error("scope failed");
  };
  await expect(execute(s)).rejects.toThrow("scope failed");
  expect(s.tensors.every((t) => t.disposed)).toBe(true);
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
});
test("sync failure releases detached CPU output", async () => {
  const s = await setup();
  s.device.sync.mockRejectedValue(new Error("sync failed"));
  await expect(execute(s)).rejects.toThrow("sync failed");
  expect(s.tensors.every((t) => t.disposed)).toBe(true);
});

test("rejects a malformed cached parameter", async () => {
  const s = await setup();
  s.runtime.tensorCacheGet = () => new Tensor([2]);
  await expect(execute(s)).rejects.toThrow("Cached parameter mismatch");
  expect(s.calls).toHaveLength(0);
});

test("rejects a nonfinite embedding after sync", async () => {
  const s = await setup();
  s.device.sync.mockImplementation(async () => {
    for (const tensor of s.tensors) tensor.data.fill(NaN);
  });
  await expect(execute(s)).rejects.toThrow("Invalid embedding output");
});

test("rejects incompatible embedding shape", async () => {
  const s = await setup(),
    original = s.funcs.forward;
  s.funcs.forward = (x, p) => {
    const value = original(x, p);
    value.shape = [2];
    return value;
  };
  await expect(execute(s)).rejects.toThrow("unexpected shape");
});

test("rejects invalid compiled minimum and embedding dimensions", async () => {
  for (const change of [
    { min_mel_frames: 4 },
    { min_mel_frames: 33 },
    { embedding_dim: 4 },
  ]) {
    const s = await setup();
    Object.assign(s.metadata.reference_conditioner, change);
    await expect(execute(s)).rejects.toThrow("metadata");
  }
});

import { verifyPendingTVMDisposal } from "./tvm_lifecycle_test_utils";
test.each([false, true])(
  "pending GPU disposal preserves resources and sync error (%s)",
  async (fail) => {
    const s = await setup();
    await verifyPendingTVMDisposal(s, "audio", input(), fail);
  },
);
