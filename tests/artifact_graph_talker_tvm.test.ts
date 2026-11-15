import { webcrypto } from "crypto";
import type * as tvmjs from "@mlc-ai/web-runtime";
import {
  createTalkerTVMFactory,
  TalkerTVMOptions,
} from "../src/artifact_graph_talker_tvm";
import {
  computeGraphInterfaceId,
  parseModelPackageManifestV2,
  parseCompiledProgramArtifactV2,
} from "../src/artifact_manifest_v2";
import { loadComponent } from "../src/artifact_graph_tvm";
import { GraphInputEvent } from "../src/artifact_graph_executor";
import { verifyExceptionalTVMReadback } from "./tvm_lifecycle_test_utils";
jest.mock("../src/artifact_graph_tvm", () => ({
  ...jest.requireActual("../src/artifact_graph_tvm"),
  loadComponent: jest.fn(),
}));
beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  }),
);
class Tensor {
  data: Float32Array;
  constructor(
    public shape: number[],
    public dtype = "float32",
  ) {
    this.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  }
  copyFrom(x: Tensor | Float32Array | Int32Array) {
    this.data.set(x instanceof Tensor ? x.data : x);
    return this;
  }
  toArray() {
    return this.data.slice();
  }
  view(shape: number[]) {
    const t = new Tensor(shape, this.dtype);
    t.data = this.data;
    return t;
  }
  dispose = jest.fn();
}
async function setup(
  tokens = [4, 5, 8294],
  maxCodes = 4,
  sampling?: TalkerTVMOptions["sampling"],
) {
  const prefix = {
    semantic: "audio.talker_prefix.v1",
    dtype: "float32",
    shape: ["prefix", 2048],
    delivery: "value",
  };
  const reply = {
    ...prefix,
    semantic: "audio.talker_reply.v1",
    shape: ["reply", 2048],
  };
  const code = {
    semantic: "audio.codec_ids.v1",
    dtype: "int32",
    shape: ["codes"],
    delivery: "value",
  };
  const spec = {
    kind: "autoregressive",
    contract: "qwen2_5_omni.talker.v1",
    inputs: { prefix_embeddings: prefix, reply_embeddings: reply },
    outputs: { code },
    weights: {
      manifest: "tensor-cache.json",
      parameter_schema_id: "sha256:" + "0".repeat(64),
    },
    state: { kind: "kv_cache", scope: "request" },
  };
  const model = {
    schema: "mlc.model-package",
    schema_version: 2,
    interface_id: "sha256:" + "0".repeat(64),
    components: { talker: spec },
    tasks: {
      generate: {
        inputs: Object.fromEntries(
          Object.entries(spec.inputs).map(([port, spec]) => [
            port,
            { spec, targets: [{ component: "talker", port }] },
          ]),
        ),
        outputs: {
          code: {
            spec: code,
            source: { component: "talker", port: "code" },
            optional: false,
          },
        },
        edges: [],
      },
    },
  };
  model.interface_id = await computeGraphInterfaceId(model);
  const exports = {
    run: "prefill",
    embed: "embed",
    project: "project",
    prefill: "prefill",
    decode: "decode",
    create_state: "create_state",
    reset_state: "reset_state",
  };
  const component = {
    spec,
    module: "talker.wasm",
    exports,
    resources: {
      required_features: [],
      max_storage_buffer_binding_size: 0,
      estimated_device_memory_bytes: 0,
    },
  };
  const compiled = {
    schema: "mlc.compiled-program",
    schema_version: 2,
    interface_id: model.interface_id,
    components: { talker: component },
  };
  const config = {
    model_dtype: "float32",
    embedding_size: 2048,
    hidden_size: 896,
    vocab_size: 8448,
    codec_bos_token_id: 8293,
    codec_pad_token_id: 8292,
    codec_eos_token_id: 8294,
    context_window_size: 32,
    prefill_chunk_size: 2,
  };
  const calls: { name: string; data?: number[]; shape?: number[] }[] = [];
  let position = 0;
  const faults = { forward: false, end: false, dispose: false };
  const cache = { dispose: jest.fn() };
  const logits = () => {
    if (faults.forward) throw Error("original forward failure");
    const t = new Tensor([1, 1, 8448]);
    t.data[tokens[Math.min(position++, tokens.length - 1)]] = 5;
    t.data[8293] = 10;
    return { get: () => t };
  };
  const funcs: Record<string, (...x: any[]) => any> = {
    create_state: () => cache,
    reset_state: () => calls.push({ name: "reset" }),
    embed: (x: Tensor) => {
      calls.push({ name: "embed", data: Array.from(x.data) });
      const t = new Tensor([x.shape[0], 2048]);
      for (let i = 0; i < x.shape[0]; i++)
        t.data.fill(x.data[i], i * 2048, (i + 1) * 2048);
      return t;
    },
    project: (x: Tensor) => {
      calls.push({ name: "project", data: Array.from(x.data), shape: x.shape });
      return new Tensor([x.shape[0], 896]);
    },
    prefill: (x: Tensor) => {
      calls.push({ name: "prefill", shape: x.shape });
      return logits();
    },
    decode: (x: Tensor) => {
      calls.push({ name: "decode", shape: x.shape });
      return logits();
    },
  };
  const vm = {
    getFunction: (name: string) => Object.assign(funcs[name], { dispose() {} }),
  };
  (loadComponent as jest.Mock).mockResolvedValue({
    vm,
    metadata: { talker: config },
    params: [],
  });
  const cpu = {},
    cpuTensors: Tensor[] = [];
  const runtime = {
    beginScope() {},
    endScope() {},
    detachFromCurrentScope: <T>(v: T) => v,
    getParamsFromCacheByName: () => ({ dispose() {} }),
    getGlobalFunc: (name: string) =>
      Object.assign(
        () => {
          if (name.endsWith("end_forward") && faults.end)
            throw Error("cache cleanup failed");
        },
        { dispose() {} },
      ),
    makeShapeTuple: (v: number[]) => v,
    scalar: (v: number) => v,
    cpu: () => cpu,
    empty: (shape: number[], dtype: string, target: unknown) => {
      const t = new Tensor(shape, dtype);
      if (target === cpu) {
        cpuTensors.push(t);
        t.dispose.mockImplementation(() => {
          if (faults.dispose) throw Error("CPU cleanup failed");
        });
      }
      return t;
    },
  };
  const device = { sync: jest.fn(async () => {}) };
  const factory = await createTalkerTVMFactory({
    runtime: runtime as unknown as tvmjs.Instance,
    device: device as unknown as tvmjs.DLDevice,
    model: parseModelPackageManifestV2(model),
    compiled: parseCompiledProgramArtifactV2(compiled),
    componentId: "talker",
    contextWindowSize: 32,
    maxCodes,
    sampling,
    capabilities: { features: new Set(), maxStorageBufferBindingSize: 1e9 },
  });
  const controller = new AbortController();
  const b = await factory.create("talker", component as any, controller.signal);
  return {
    b,
    calls,
    config,
    device,
    cache,
    factory,
    controller,
    component,
    faults,
    cpuTensors,
    runtime,
  };
}
async function* input(
  prefixRows = 2,
  replyRows = 1,
): AsyncIterable<GraphInputEvent> {
  for (const [port, rows, value] of [
    ["prefix_embeddings", prefixRows, 0.25],
    ["reply_embeddings", replyRows, 0.5],
  ] as const) {
    yield {
      kind: "data",
      port,
      tensor: {
        shape: [rows, 2048],
        data: new Float32Array(rows * 2048).fill(value),
      },
    };
    yield { kind: "end", port };
  }
}
async function collect(s: Awaited<ReturnType<typeof setup>>, events = input()) {
  const result = [];
  for await (const event of s.b.run(events, s.controller.signal))
    result.push(...event.tensor.data);
  return result;
}
test("BOS-only suppression, pad/BOS additions, shifted reply and stripped EOS", async () => {
  const s = await setup();
  expect(await collect(s)).toEqual([4, 5]);
  const projects = s.calls.filter((x) => x.name === "project");
  expect(projects[0].data![0]).toBe(8292.25);
  expect(projects[0].data![2048]).toBe(8293.25);
  expect(projects[1].data![0]).toBe(4.5);
  expect(projects[2].data![0]).toBe(5.5);
});
test("PAD terminates without entering DiT codes", async () => {
  const s = await setup([3, 8292]);
  expect(await collect(s)).toEqual([3]);
});
test.each([8194, 8296, 8447])(
  "reject unsupported selected codec %i",
  async (token) => {
    const s = await setup([token]);
    await expect(collect(s)).rejects.toThrow("unsupported codec");
  },
);
test("empty EOS result rejects explicitly", async () => {
  const s = await setup([8294]);
  await expect(collect(s)).rejects.toThrow("no codec");
});
test("maxCodes bounds generation without an extra decode", async () => {
  const s = await setup([6], 2);
  expect(await collect(s)).toEqual([6, 6]);
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(1);
});
test.each([0, 257, 1.5])("reject invalid maxCodes %s", async (n) => {
  await expect(setup([1], n)).rejects.toThrow("limits");
});
test("context rejects before forward", async () => {
  const s = await setup();
  await expect(collect(s, input(31))).rejects.toThrow("context");
  expect(s.calls.some((x) => x.name === "project")).toBe(false);
});
test("chunked prefix preserves final-two specials", async () => {
  const s = await setup([4, 5, 8294]);
  expect(await collect(s, input(3))).toEqual([5]);
  const p = s.calls.filter((x) => x.name === "project");
  expect(p[0].data![0]).toBe(0.25);
  expect(p[0].data![2048]).toBe(8292.25);
  expect(p[1].data![0]).toBe(8293.25);
});
test("aborted synchronization stops before forward and disposes CPU tensor", async () => {
  const s = await setup();
  s.device.sync.mockImplementation(async () => {
    s.controller.abort();
  });
  await expect(collect(s)).rejects.toThrow("aborted");
  expect(s.calls.some((x) => x.name === "project")).toBe(false);
  expect(s.cpuTensors[0].dispose).toHaveBeenCalled();
});
test("synchronization failure survives CPU disposal failure", async () => {
  const s = await setup();
  s.faults.dispose = true;
  s.device.sync.mockRejectedValue(Error("sync failed"));
  await expect(collect(s)).rejects.toThrow("sync failed");
});
test("forward failure survives cache cleanup failure", async () => {
  const s = await setup();
  s.faults.forward = true;
  s.faults.end = true;
  await expect(collect(s)).rejects.toThrow("original forward failure");
});
test("factory exclusivity and disposal release", async () => {
  const s = await setup();
  await expect(
    s.factory.create("talker", s.component as any, s.controller.signal),
  ).rejects.toThrow("already in use");
  await s.b.dispose();
  const next = await s.factory.create(
    "talker",
    s.component as any,
    s.controller.signal,
  );
  await next.dispose();
});
test("nonfinite input rejected before embedding", async () => {
  const s = await setup();
  async function* bad() {
    for await (const e of input()) {
      if (e.kind === "data") e.tensor.data[0] = NaN;
      yield e;
    }
  }
  await expect(collect(s, bad())).rejects.toThrow("conditioning");
  expect(s.calls.some((x) => x.name === "embed")).toBe(false);
});

test("reply rows advance and repeat only the final row", async () => {
  const s = await setup([4, 5, 6, 8294]);
  async function* rows() {
    for await (const event of input(2, 2)) {
      if (event.kind === "data" && event.port === "reply_embeddings") {
        event.tensor.data.fill(1, 0, 2048);
        event.tensor.data.fill(2, 2048);
      }
      yield event;
    }
  }
  expect(await collect(s, rows())).toEqual([4, 5, 6]);
  expect(
    s.calls
      .filter((x) => x.name === "project")
      .slice(1)
      .map((x) => x.data![0]),
  ).toEqual([5, 7, 8]);
});
test("input EOF is mandatory", async () => {
  const s = await setup();
  async function* missing() {
    for await (const e of input()) if (e.kind !== "end") yield e;
  }
  await expect(collect(s, missing())).rejects.toThrow("EOF");
});
test("backend cannot reuse completed request state", async () => {
  const s = await setup();
  await collect(s);
  await expect(collect(s)).rejects.toThrow("once");
});

test("dispose waits for pending GPU readback before releasing request state", async () => {
  const s = await setup();
  let release!: () => void, entered!: () => void;
  const started = new Promise<void>((resolve) => {
    entered = resolve;
  });
  s.device.sync.mockImplementation(() => {
    entered();
    return new Promise<void>((resolve) => {
      release = resolve;
    });
  });
  const result = collect(s);
  const rejected = expect(result).rejects.toThrow("disposed");
  await started;
  let complete = false;
  const disposal = Promise.resolve(s.b.dispose()).then(() => {
    complete = true;
  });
  await Promise.resolve();
  expect(complete).toBe(false);
  expect(s.cache.dispose).not.toHaveBeenCalled();
  await expect(
    s.factory.create("talker", s.component as any, s.controller.signal),
  ).rejects.toThrow("already in use");
  release();
  await rejected;
  await disposal;
  expect(s.cache.dispose).toHaveBeenCalledTimes(1);
  expect(s.cpuTensors[0].dispose).toHaveBeenCalledTimes(1);
  const next = await s.factory.create(
    "talker",
    s.component as any,
    s.controller.signal,
  );
  await next.dispose();
});

test.each(["scope", "copy", "microtask"] as const)(
  "queued readback survives %s failure",
  async (kind) => {
    const s = await setup();
    await verifyExceptionalTVMReadback(
      { ...s, component: s.component },
      "talker",
      input(),
      kind,
    );
  },
);

const samplingOptions = () => ({
  temperature: 0.9,
  topK: 1,
  topP: 0.8,
  repetitionPenalty: 1.05,
  random: jest.fn(() => 0.5),
});
test("optional sampling preserves EOS handling and snapshots scalar settings", async () => {
  const sampling = samplingOptions();
  const s = await setup([4, 5, 8294], 4, sampling);
  sampling.temperature = -1;
  expect(await collect(s)).toEqual([4, 5]);
  expect(sampling.random).toHaveBeenCalledTimes(3);
});
test("invalid sampling rejects before loading a component", async () => {
  await expect(
    setup([4], 4, { ...samplingOptions(), temperature: 0 }),
  ).rejects.toThrow("Invalid Talker sampling");
});
test("invalid categorical draw cleans up request state", async () => {
  const s = await setup([4], 4, { ...samplingOptions(), random: () => 1 });
  await expect(collect(s)).rejects.toThrow();
  await s.b.dispose();
  expect(s.cache.dispose).toHaveBeenCalledTimes(1);
});

test("sampling sees codec mask prefix and the complete generated history", async () => {
  const sampler = await import("../src/artifact_graph_talker_sampling");
  const original = sampler.sampleTalkerToken;
  const histories: number[][] = [];
  const spy = jest
    .spyOn(sampler, "sampleTalkerToken")
    .mockImplementation((logits, history, options) => {
      histories.push([...history]);
      return original(logits, history, options);
    });
  try {
    const s = await setup([4, 4, 5, 8294], 4, samplingOptions());
    expect(await collect(s, input(3))).toEqual([4, 5]);
    expect(histories).toEqual([
      [8296, 8292, 8293],
      [8296, 8292, 8293, 4],
      [8296, 8292, 8293, 4, 5],
    ]);
    await s.b.dispose();
  } finally {
    spy.mockRestore();
  }
});
