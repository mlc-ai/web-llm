import { createHash, webcrypto } from "crypto";
import type * as tvmjs from "@mlc-ai/web-runtime";
import {
  createFlowPredictorTVMFactory,
  integrateFlowRK4,
  FlowSamplingOptions,
} from "../src/artifact_graph_flow_tvm";
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
  const port = (
    semantic: string,
    shape: (number | string)[],
    dtype: "float32" | "int32" = "float32",
  ) => ({ semantic, dtype, shape, delivery: "value" as const });
  const inputs = {
    noise: port("audio.flow_state.v1", ["frames", 2]),
    time: port("flow.time.v1", [1]),
    code: port("audio.codec_ids.v1", ["codes"], "int32"),
    reference: port("audio.reference_embedding.v1", [3]),
    speaker: port("audio.speaker_embedding.v1", [3]),
  };
  const output = port("audio.flow_velocity.v1", ["frames", 2]);
  const spec = {
    kind: "transform" as const,
    contract: "qwen2_5_omni.flow_predictor.v1",
    inputs,
    outputs: { velocity: output },
    weights: { manifest: "tensor-cache.json", parameter_schema_id: schema },
    state: { kind: "none" as const, scope: "request" as const },
  };
  const model: ModelPackageManifestV2 = {
    schema: "mlc.model-package",
    schema_version: 2,
    interface_id: "sha256:" + "0".repeat(64),
    components: { audio: spec },
    tasks: {
      predict: {
        inputs: Object.fromEntries(
          Object.entries(inputs).map(([name, spec]) => [
            name,
            { spec, targets: [{ component: "audio", port: name }] },
          ]),
        ),
        outputs: {
          velocity: {
            spec: output,
            source: { component: "audio", port: "velocity" },
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
    flow_predictor: {
      mel_dim: 2,
      reference_dim: 3,
      speaker_dim: 3,
      repeats: 2,
      num_embeds: 17,
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
    forward: (
      x: Tensor,
      time: Tensor,
      code: Tensor,
      reference: Tensor,
      speaker: Tensor,
      p: unknown,
    ) => {
      expect(time.shape).toEqual([1]);
      expect(code.shape).toEqual([x.shape[0] / 2]);
      expect(reference.shape).toEqual([3]);
      expect(speaker.shape).toEqual([3]);
      expect(p).toBe(packed);
      calls.push({ name: "forward", shape: x.shape, data: Array.from(x.data) });
      const out = track(new Tensor(x.shape));
      out.data.fill(0.25);
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
  const factory = await createFlowPredictorTVMFactory(options);
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
  };
}

function tensors(
  frames = 4,
): Record<string, { shape: number[]; data: Float32Array | Int32Array }> {
  return {
    noise: { shape: [frames, 2], data: new Float32Array(frames * 2) },
    time: { shape: [1], data: new Float32Array([0.5]) },
    code: { shape: [frames / 2], data: new Int32Array(frames / 2).fill(1) },
    reference: { shape: [3], data: new Float32Array([1, 2, 3]) },
    speaker: { shape: [3], data: new Float32Array(3) },
  };
}
async function* events(data = tensors()): AsyncIterable<GraphInputEvent> {
  for (const [port, tensor] of Object.entries(data)) {
    yield { kind: "data", port, tensor };
    yield { kind: "end", port };
  }
}
async function execute(
  s: Awaited<ReturnType<typeof setup>>,
  data = tensors(),
  signal = new AbortController().signal,
) {
  const b = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    signal,
  );
  try {
    const out = [];
    for await (const e of b.run(events(data), signal)) out.push(e);
    return out;
  } finally {
    await b.dispose();
  }
}
test("invokes five explicit inputs and returns same-shape velocity", async () => {
  const s = await setup(),
    out = await execute(s);
  expect(out[0].tensor.shape).toEqual([4, 2]);
  expect(Array.from(out[0].tensor.data)).toEqual(Array(8).fill(0.25));
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
  await execute(s);
});
test.each([
  "frames",
  "code_count",
  "code_range",
  "time",
  "dtype",
  "nan",
  "missing",
])("rejects %s before invocation", async (kind) => {
  const s = await setup(),
    d = tensors();
  if (kind === "frames") d.noise.shape = [3, 2];
  if (kind === "code_count") d.code = { shape: [1], data: new Int32Array(1) };
  if (kind === "code_range") d.code.data[0] = 18;
  if (kind === "time") d.time.data[0] = 1.1;
  if (kind === "dtype") d.noise.data = new Int32Array(8);
  if (kind === "nan") d.reference.data[0] = NaN;
  if (kind === "missing") delete d.speaker;
  await expect(execute(s, d)).rejects.toThrow();
  expect(s.calls).toHaveLength(0);
});
test("scope failure releases detached output", async () => {
  const s = await setup(),
    end = s.runtime.endScope;
  s.runtime.endScope = () => {
    end();
    if (s.calls.length) throw Error("scope failure");
  };
  await expect(execute(s)).rejects.toThrow("scope failure");
  expect(s.tensors.every((t) => t.disposed)).toBe(true);
});
test("cancellation after sync releases detached output", async () => {
  const s = await setup(),
    c = new AbortController();
  s.device.sync.mockImplementation(async () => {
    c.abort();
  });
  await expect(execute(s, tensors(), c.signal)).rejects.toThrow("aborted");
  expect(s.tensors.every((t) => t.disposed)).toBe(true);
});
function sampling(): FlowSamplingOptions {
  const d = tensors();
  return {
    ...d,
    noise: d.noise,
    code: d.code,
    reference: d.reference,
    speaker: d.speaker,
    nullReference: { shape: [3], data: new Float32Array(3) },
    maxFrames: 32,
    melDim: 2,
    referenceDim: 3,
    speakerDim: 3,
    repeats: 2,
    numEmbeds: 17,
    timePoints: new Float32Array([0, 0.5, 1]),
    guidanceScale: 0.5,
    maxTimePoints: 4,
  };
}
test("RK4 preserves constant conditioned guidance and exact stage times", async () => {
  const times: number[] = [];
  const out = await integrateFlowRK4(
    async (x) => {
      times.push(x.time.data[0]);
      return {
        shape: [4, 2],
        data: new Float32Array(8).fill(x.code.data[0] === 0 ? 1 : 3),
      };
    },
    sampling(),
    new AbortController().signal,
  );
  expect(Array.from(out.data)).toEqual(Array(8).fill(4));
  expect(times).toHaveLength(16);
  expect(times[2]).toBeCloseTo(1 / 6, 7);
  expect(times[4]).toBeCloseTo(1 / 3, 7);
  expect(times[6]).toBe(0.5);
});
test.each([
  [0, "shape"],
  [1, "shape"],
  [0, "dtype"],
  [1, "nan"],
] as const)(
  "rejects bad CFG call %s/%s before broadcasting",
  async (bad, kind) => {
    let calls = 0;
    await expect(
      integrateFlowRK4(
        async () => {
          const fail = calls++ === bad;
          return {
            shape: fail && kind === "shape" ? [1, 2] : [4, 2],
            data:
              fail && kind === "dtype"
                ? new Int32Array(8)
                : new Float32Array(fail && kind === "shape" ? 2 : 8).fill(
                    fail && kind === "nan" ? NaN : 1,
                  ),
          };
        },
        sampling(),
        new AbortController().signal,
      ),
    ).rejects.toThrow("velocity");
    expect(calls).toBe(bad + 1);
  },
);
test("abort during prediction prevents next CFG call", async () => {
  const c = new AbortController();
  let calls = 0;
  await expect(
    integrateFlowRK4(
      async () => {
        calls++;
        c.abort();
        return { shape: [4, 2], data: new Float32Array(8) };
      },
      sampling(),
      c.signal,
    ),
  ).rejects.toThrow("aborted");
  expect(calls).toBe(1);
});
test.each(["grid", "limit", "null"])(
  "rejects invalid solver %s",
  async (kind) => {
    const o = sampling();
    if (kind === "grid") o.timePoints = new Float32Array([0, 0.5, 0.4]);
    if (kind === "limit") o.maxTimePoints = 2;
    if (kind === "null") o.nullReference.shape = [1];
    const fn = jest.fn();
    await expect(
      integrateFlowRK4(fn, o, new AbortController().signal),
    ).rejects.toThrow();
    expect(fn).not.toHaveBeenCalled();
  },
);

test("snapshots caller inputs while prediction is pending", async () => {
  const original = sampling();
  let resume!: () => void;
  const paused = new Promise<void>((resolve) => {
    resume = resolve;
  });
  let calls = 0;
  const result = integrateFlowRK4(
    async (inputs) => {
      if (calls++ === 0) await paused;
      for (const value of Object.values(inputs))
        expect(Array.from(value.data).every(Number.isFinite)).toBe(true);
      expect(inputs.reference.shape).toEqual([3]);
      expect(inputs.speaker.shape).toEqual([3]);
      return {
        shape: [4, 2],
        data: new Float32Array(8).fill(inputs.code.data[0] === 0 ? 1 : 3),
      };
    },
    original,
    new AbortController().signal,
  );
  original.timePoints.fill(NaN);
  original.code.data.fill(0);
  original.reference.data.fill(NaN);
  original.nullReference.data.fill(NaN);
  original.speaker.data.fill(NaN);
  original.noise.data.fill(NaN);
  original.reference.shape[0] = 1;
  original.speaker.shape[0] = 1;
  original.guidanceScale = 0;
  original.maxTimePoints = 0;
  resume();
  expect(Array.from((await result).data)).toEqual(Array(8).fill(4));
  expect(calls).toBe(16);
});

import { verifyPendingTVMDisposal } from "./tvm_lifecycle_test_utils";
test.each([false, true])(
  "pending GPU disposal preserves resources and sync error (%s)",
  async (fail) => {
    const s = await setup();
    await verifyPendingTVMDisposal(s, "audio", events(), fail);
  },
);
