import { createHash, webcrypto } from "crypto";
import type * as tvmjs from "@mlc-ai/web-runtime";
import {
  createAudioEncoderTVMFactory,
  createWhisperAudioTVMFactory,
} from "../src/artifact_graph_tvm";
import {
  computeGraphInterfaceId,
  ModelPackageManifestV2,
  CompiledProgramArtifactV2,
} from "../src/artifact_manifest_v2";
import { GraphInputEvent } from "../src/artifact_graph_executor";
import {
  verifyExceptionalTVMReadback,
  verifyPendingTVMDisposal,
} from "./tvm_lifecycle_test_utils";
beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  }),
);
async function fixture(processor = false) {
  const schema =
    "sha256:" +
    createHash("sha256")
      .update('[{"dtype":"float32","name":"weight","shape":[1]}]')
      .digest("hex");
  const input = {
    semantic: "qwen.mel.v1",
    dtype: "float32" as const,
    shape: [2, "frames"],
    delivery: "value" as const,
  };
  const output = {
    semantic: "qwen.embeddings.v1",
    dtype: "float32" as const,
    shape: ["tokens", 3],
    delivery: "value" as const,
  };
  const spec = {
    kind: "encoder" as const,
    contract: "qwen2_5_omni.audio_encoder.v1",
    inputs: { mel_features: input },
    outputs: { embeddings: output },
    weights: { manifest: "tensor-cache.json", parameter_schema_id: schema },
    state: { kind: "none" as const, scope: "request" as const },
  };
  const model: ModelPackageManifestV2 = {
    schema: "mlc.model-package",
    schema_version: 2,
    interface_id: "sha256:" + "0".repeat(64),
    components: { audio: spec },
    tasks: {
      encode: {
        inputs: {
          mel: {
            spec: input,
            targets: [{ component: "audio", port: "mel_features" }],
          },
        },
        outputs: {
          embeddings: {
            spec: output,
            source: { component: "audio", port: "embeddings" },
            optional: false,
          },
        },
        edges: [],
      },
    },
  };
  if (processor) {
    const pcm = {
      semantic: "audio.pcm.v1",
      dtype: "float32" as const,
      shape: ["samples"],
      delivery: "value" as const,
    };
    const mel = {
      semantic: "audio.log_mel.v1",
      dtype: "float32" as const,
      shape: [128, "frames"],
      delivery: "value" as const,
    };
    model.components.audio = {
      ...spec,
      kind: "transform",
      contract: "qwen2_5_omni.audio_features.v1",
      inputs: { pcm },
      outputs: { mel_features: mel },
    };
    model.tasks.encode = {
      inputs: {
        pcm: { spec: pcm, targets: [{ component: "audio", port: "pcm" }] },
      },
      outputs: {
        mel_features: {
          spec: mel,
          source: { component: "audio", port: "mel_features" },
          optional: false,
        },
      },
      edges: [],
    };
  }
  model.interface_id = await computeGraphInterfaceId(model);
  const compiled: CompiledProgramArtifactV2 = {
    schema: "mlc.compiled-program",
    schema_version: 2,
    interface_id: model.interface_id,
    components: {
      audio: {
        spec: model.components.audio,
        module: "audio.wasm",
        exports: {
          run: "encode_chunk",
          encode_chunk: "encode_chunk",
          project: "project",
        },
        resources: {
          required_features: [],
          max_storage_buffer_binding_size: 0,
          estimated_device_memory_bytes: 0,
        },
      },
    },
  };
  if (processor)
    compiled.components.audio.exports = { run: "forward", forward: "forward" };
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
async function setup(processor = false) {
  const { model, compiled } = await fixture(processor);
  const metadata = {
    artifact: compiled,
    params: [{ name: "weight", shape: [1], dtype: "float32" }],
    audio_encoder: { hidden_size: 2, n_window: 100, max_pooled_frames: 7500 },
    audio_processor: {
      num_samples: 640,
      sampling_rate: 16000,
      hop_length: 160,
      num_mel_bins: 128,
    },
  };
  const calls: { name: string; shape: number[]; data: number[] }[] = [];
  const packed = { dispose: jest.fn() };
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
      const out = track(new Tensor([128, 4]));
      out.data = Float32Array.from({ length: 512 }, (_, i) => i);
      return out;
    },
    encode_chunk: (x: Tensor, valid: Tensor, p: unknown) => {
      expect(valid.shape).toEqual([1]);
      expect(p).toBe(packed);
      calls.push({ name: "encode", shape: x.shape, data: Array.from(x.data) });
      const out = track(new Tensor([Math.ceil(x.shape[2] / 2), 2]));
      for (let i = 0; i < out.data.length; i++) out.data[i] = i;
      return out;
    },
    project: (x: Tensor, p: unknown) => {
      expect(p).toBe(packed);
      calls.push({ name: "project", shape: x.shape, data: Array.from(x.data) });
      return track(new Tensor([x.shape[0], 3]));
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
    empty: (shape: number[], dtype: string) => track(new Tensor(shape, dtype)),
    cpu: () => ({}),
  };
  const factory = await (
    processor ? createWhisperAudioTVMFactory : createAudioEncoderTVMFactory
  )({
    runtime: runtime as unknown as tvmjs.Instance,
    device: device as unknown as tvmjs.DLDevice,
    model,
    compiled,
    componentId: "audio",
    maxFrames: 500,
    capabilities: { features: new Set(), maxStorageBufferBindingSize: 1024 },
  });
  return { factory, compiled, metadata, calls, device, vm, packed, runtime };
}
async function* input(frames: number): AsyncIterable<GraphInputEvent> {
  yield {
    kind: "data",
    port: "mel_features",
    tensor: {
      shape: [2, frames],
      data: Float32Array.from({ length: 2 * frames }, (_, i) => i),
    },
  };
  yield { kind: "end", port: "mel_features" };
}
test("real ABI calls preserve channel-major chunk slices and pool once per clip", async () => {
  const s = await setup();
  const b = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    new AbortController().signal,
  );
  const outputs = [];
  for await (const x of b.run(input(201), new AbortController().signal))
    outputs.push(x);
  expect(s.calls.map((c) => [c.name, c.shape])).toEqual([
    ["encode", [1, 2, 200]],
    ["encode", [1, 2, 200]],
    ["project", [50, 2, 2]],
  ]);
  expect(s.calls[0].data[200]).toBe(201);
  expect(s.calls[1].data[0]).toBe(200);
  expect(s.calls[1].data[200]).toBe(401);
  expect(s.calls[1].data[1]).toBe(0);
  expect(outputs[0].tensor.shape).toEqual([50, 3]);
  expect(s.device.sync).toHaveBeenCalledTimes(3);
  await b.dispose();
  await b.dispose();
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
  expect(s.packed.dispose).toHaveBeenCalledTimes(1);
});
test.each([1, 2])(
  "short clip %i yields empty without GPU invocation",
  async (frames) => {
    const s = await setup();
    const b = await s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    );
    const result = [];
    for await (const x of b.run(input(frames), new AbortController().signal))
      result.push(x);
    expect(result[0].tensor.shape).toEqual([0, 3]);
    expect(s.calls).toEqual([]);
    await b.dispose();
  },
);
test("metadata mismatch fails before parameter binding and releases VM", async () => {
  const s = await setup();
  s.metadata.params[0].shape = [2];
  await expect(
    s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    ),
  ).rejects.toThrow("schema mismatch");
  expect(s.runtime.getParamsFromCacheByName).not.toHaveBeenCalled();
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
});
test("exclusive factory and abort before upload", async () => {
  const s = await setup();
  const c = new AbortController();
  const b = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    c.signal,
  );
  await expect(
    s.factory.create("audio", s.compiled.components.audio, c.signal),
  ).rejects.toThrow("already in use");
  c.abort();
  await expect(
    b.run(input(201), c.signal)[Symbol.asyncIterator]().next(),
  ).rejects.toThrow("aborted");
  expect(s.calls).toEqual([]);
  await b.dispose();
});
test("oversized clip fails before GPU allocation", async () => {
  const s = await setup();
  const b = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    new AbortController().signal,
  );
  await expect(
    b
      .run(input(501), new AbortController().signal)
      [Symbol.asyncIterator]()
      .next(),
  ).rejects.toThrow("Invalid mel");
  expect(s.calls).toEqual([]);
  await b.dispose();
});

test("abort during device synchronization prevents output delivery", async () => {
  const s = await setup();
  const controller = new AbortController();
  s.device.sync.mockImplementation(async () => {
    controller.abort();
  });
  const backend = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    controller.signal,
  );
  await expect(
    backend.run(input(201), controller.signal)[Symbol.asyncIterator]().next(),
  ).rejects.toThrow("aborted");
  expect(s.calls).toHaveLength(1);
  await backend.dispose();
});

test("VM artifact mismatch fails before binding", async () => {
  const s = await setup();
  s.metadata.artifact = JSON.parse(JSON.stringify(s.compiled));
  s.metadata.artifact.components.audio.module = "wrong.wasm";
  await expect(
    s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    ),
  ).rejects.toThrow("VM metadata");
  expect(s.runtime.getParamsFromCacheByName).not.toHaveBeenCalled();
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
});

test("run cancellation is respected even when distinct from factory signal", async () => {
  const s = await setup();
  const backend = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    new AbortController().signal,
  );
  const controller = new AbortController();
  controller.abort();
  await expect(
    backend.run(input(200), controller.signal)[Symbol.asyncIterator]().next(),
  ).rejects.toThrow("aborted");
  expect(s.calls).toHaveLength(0);
  await backend.dispose();
});

test("compiled projection bound limits accepted clip capacity", async () => {
  const s = await setup();
  s.metadata.audio_encoder.max_pooled_frames = 1;
  await expect(
    s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    ),
  ).rejects.toThrow("compiled projection bound");
  expect(s.runtime.getParamsFromCacheByName).not.toHaveBeenCalled();
});

async function* pcmInput(length: number): AsyncIterable<GraphInputEvent> {
  yield {
    kind: "data",
    port: "pcm",
    tensor: { shape: [length], data: new Float32Array(length).fill(0.25) },
  };
  yield { kind: "end", port: "pcm" };
}
test("PCM binding zero-pads once and crops each mel row using ceiling", async () => {
  const s = await setup(true);
  const b = await s.factory.create(
    "audio",
    s.compiled.components.audio,
    new AbortController().signal,
  );
  const rows = [];
  for await (const out of b.run(pcmInput(161), new AbortController().signal))
    rows.push(out);
  expect(s.calls[0].shape).toEqual([640]);
  expect(s.calls[0].data[160]).toBe(0.25);
  expect(s.calls[0].data[161]).toBe(0);
  expect(rows[0].tensor.shape).toEqual([128, 2]);
  expect(Array.from(rows[0].tensor.data).slice(0, 6)).toEqual([
    0, 1, 4, 5, 8, 9,
  ]);
  await b.dispose();
  expect(s.vm.dispose).toHaveBeenCalledTimes(1);
});
test.each([0, 641])(
  "PCM rejects unsupported length %i before GPU call",
  async (length) => {
    const s = await setup(true);
    const b = await s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    );
    await expect(
      b
        .run(pcmInput(length), new AbortController().signal)
        [Symbol.asyncIterator]()
        .next(),
    ).rejects.toThrow("Invalid PCM");
    expect(s.calls).toHaveLength(0);
    await b.dispose();
  },
);
test("PCM rejects incompatible processor metadata before binding weights", async () => {
  const s = await setup(true);
  s.metadata.audio_processor.sampling_rate = 48000;
  await expect(
    s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    ),
  ).rejects.toThrow("processor metadata");
  expect(s.runtime.getParamsFromCacheByName).not.toHaveBeenCalled();
});

test.each([false, true])(
  "pending GPU disposal preserves audio VM (%s)",
  async (fail) => {
    const s = await setup();
    await verifyPendingTVMDisposal(s, "audio", input(201), fail);
  },
);
test.each([false, true])(
  "pending GPU disposal preserves PCM VM (%s)",
  async (fail) => {
    const s = await setup(true);
    async function* pcm(): AsyncIterable<GraphInputEvent> {
      yield {
        kind: "data",
        port: "pcm",
        tensor: { shape: [400], data: new Float32Array(400) },
      };
      yield { kind: "end", port: "pcm" };
    }
    await verifyPendingTVMDisposal(s, "audio", pcm(), fail);
  },
);

test.each(["scope", "copy", "microtask"] as const)(
  "queued audio readback survives %s failure",
  async (kind) => {
    await verifyExceptionalTVMReadback(
      await setup(),
      "audio",
      input(201),
      kind,
    );
  },
);
async function* lifecyclePCM(): AsyncIterable<GraphInputEvent> {
  yield {
    kind: "data",
    port: "pcm",
    tensor: { shape: [400], data: new Float32Array(400) },
  };
  yield { kind: "end", port: "pcm" };
}
test.each(["scope", "copy", "microtask"] as const)(
  "queued PCM readback survives %s failure",
  async (kind) => {
    await verifyExceptionalTVMReadback(
      await setup(true),
      "audio",
      lifecyclePCM(),
      kind,
    );
  },
);
test.each([false, true])(
  "CPU cleanup preserves original sync error (PCM=%s)",
  async (processor) => {
    const s = await setup(processor);
    const b = await s.factory.create(
      "audio",
      s.compiled.components.audio,
      new AbortController().signal,
    );
    const cpuDevice = s.runtime.cpu();
    s.runtime.cpu = () => cpuDevice;
    const empty = s.runtime.empty;
    s.runtime.empty = ((...args: any[]) => {
      const value = (empty as any)(...args);
      if (args[2] === s.runtime.cpu())
        value.dispose = jest.fn(() => {
          throw Error("cleanup failure");
        });
      return value;
    }) as any;
    s.device.sync.mockRejectedValue(Error("original sync failure"));
    await expect(
      b
        .run(
          processor ? lifecyclePCM() : input(201),
          new AbortController().signal,
        )
        [Symbol.asyncIterator]()
        .next(),
    ).rejects.toThrow("original sync failure");
    await b.dispose();
  },
);
