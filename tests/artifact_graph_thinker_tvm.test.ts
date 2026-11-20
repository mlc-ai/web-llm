import { float32ToHalf } from "../src/artifact_graph_thinker_tvm";
import { webcrypto } from "crypto";
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createThinkerTVMFactory } from "../src/artifact_graph_thinker_tvm";
import { loadComponent } from "../src/artifact_graph_tvm";
import { GraphInputEvent } from "../src/artifact_graph_executor";
import { computeGraphInterfaceId } from "../src/artifact_manifest_v2";
import { readFileSync } from "fs";
import { join } from "path";
import {
  verifyExceptionalTVMReadback,
  verifyPendingTVMDisposal,
} from "./tvm_lifecycle_test_utils";
// Bit patterns below are IEEE 754 binary16, independent of the TVM runtime.
test.each([
  [0, 0],
  [-0, 0x8000],
  [1, 0x3c00],
  [-2, 0xc000],
  [65504, 0x7bff],
  [65520, 0x7c00],
  [Infinity, 0x7c00],
  [-Infinity, 0xfc00],
  [2 ** -24, 1],
  [2 ** -25, 0],
  [3 * 2 ** -25, 2],
  [2 ** -14, 0x400],
  [1 + 2 ** -11, 0x3c00],
  [1 + 3 * 2 ** -11, 0x3c02],
])("half conversion of %s", (value, bits) => {
  expect(float32ToHalf(value)).toBe(bits);
});
test("NaN remains NaN", () => {
  expect(float32ToHalf(NaN) & 0x7c00).toBe(0x7c00);
  expect(float32ToHalf(NaN) & 0x3ff).not.toBe(0);
});

const fixture = (name: string) =>
  JSON.parse(
    readFileSync(
      join(__dirname, "fixtures/thinker_tvm", name + ".json"),
      "utf8",
    ),
  );
const model = fixture("model"),
  compiled = fixture("compiled");
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
  copyFromRawBytes(bytes: Uint8Array) {
    this.data.set(
      new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2),
    );
    return this;
  }
  toRawBytes() {
    return new Uint8Array(Uint16Array.from(this.data).buffer);
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
async function setup(dtype = "float32", maxNewTokens = 3, qwen3 = false) {
  const calls: { name: string; data?: number[]; shape?: number[] }[] = [];
  const config = {
    model_dtype: dtype,
    hidden_size: 128,
    vocab_size: qwen3 ? 152064 : 64,
    active_vocab_size: qwen3 ? 152064 : 64,
    audio_token_id: qwen3 ? 151675 : 151646,
    image_token_id: 151655,
    video_token_id: 151656,
    eos_token_ids: qwen3 ? [151645] : [151643, 151645],
    context_window_size: 32,
    prefill_chunk_size: 2,
  };
  const cache = { dispose: jest.fn() };
  let counter = 0;
  const funcs: Record<string, (...args: any[]) => any> = {
    create_tir_paged_kv_cache: (...args) => {
      expect(args).toEqual([[1], [32], [2], [16], [0]]);
      return cache;
    },
    reset_state: () => {
      calls.push({ name: "reset" });
    },
    embed: (x: Tensor) => {
      calls.push({ name: "embed", data: Array.from(x.data) });
      return new Tensor([x.shape[0], 128], dtype);
    },
    prefill: (x: Tensor) => {
      if (faults.forward) throw new Error("original forward failed");
      calls.push({ name: "prefill", shape: x.shape, data: Array.from(x.data) });
      const t = new Tensor([1, 1, config.vocab_size]);
      t.data[++counter] = 1;
      return { get: () => t };
    },
    decode: (x: Tensor) => {
      calls.push({ name: "decode", shape: x.shape });
      const t = new Tensor([1, 1, config.vocab_size]);
      t.data[qwen3 ? config.eos_token_ids[0] : ++counter] = 1;
      return { get: () => t };
    },
  };
  const vm = {
    getFunction: (name: string) =>
      Object.assign(funcs[name], { dispose: jest.fn() }),
  };
  (loadComponent as jest.Mock).mockResolvedValue({
    vm,
    metadata: { thinker: config },
    params: [],
  });
  const cpuDevice = {};
  const cpuTensors: Tensor[] = [];
  const faults = { endAt: 0, endCalls: 0, forward: false };
  const runtime = {
    beginScope() {},
    endScope() {},
    detachFromCurrentScope: <T>(v: T) => v,
    getParamsFromCacheByName: () => ({ dispose() {} }),
    getGlobalFunc: (name: string) =>
      Object.assign(
        (...args: any[]) => {
          calls.push({ name: name.split("_").at(-1)!, data: args[2] });
          if (
            name === "vm.builtin.kv_state_end_forward" &&
            ++faults.endCalls === faults.endAt
          )
            throw new Error("cache end failed");
        },
        { dispose() {} },
      ),
    makeShapeTuple: (x: number[]) => x,
    scalar: (x: number) => x,
    empty: (shape: number[], dt: string, target: unknown) => {
      const tensor = new Tensor(shape, dt);
      if (target === cpuDevice) cpuTensors.push(tensor);
      return tensor;
    },
    cpu: () => cpuDevice,
  };
  const device = { sync: jest.fn(async () => {}) };
  const requestModel = qwen3 ? fixture("model") : model;
  const requestCompiled = qwen3 ? fixture("compiled") : compiled;
  if (qwen3) {
    const spec = requestModel.components.thinker;
    spec.contract = "qwen3_omni.thinker_audio.v1";
    requestModel.interface_id = await computeGraphInterfaceId(requestModel);
    requestCompiled.interface_id = requestModel.interface_id;
    requestCompiled.components.thinker.spec = spec;
  }
  const factory = await createThinkerTVMFactory({
    runtime: runtime as unknown as tvmjs.Instance,
    model: requestModel,
    compiled: requestCompiled,
    componentId: "thinker",
    device: device as unknown as tvmjs.DLDevice,
    contextWindowSize: 32,
    maxNewTokens,
    capabilities: { features: new Set(), maxStorageBufferBindingSize: 1e9 },
  });
  const signal = new AbortController().signal;
  const b = await factory.create(
    "thinker",
    requestCompiled.components.thinker,
    signal,
  );
  return {
    b,
    calls,
    config,
    device,
    cache,
    factory,
    signal,
    faults,
    runtime,
    cpuTensors,
  };
}
async function* input(
  ids = [3, 151646, 7],
  rows = 1,
  value = 0.25,
): AsyncIterable<GraphInputEvent> {
  yield {
    kind: "data",
    port: "token_ids",
    tensor: { shape: [ids.length], data: new Int32Array(ids) },
  };
  yield { kind: "end", port: "token_ids" };
  yield {
    kind: "data",
    port: "audio_embeddings",
    tensor: {
      shape: [rows, 128],
      data: new Float32Array(rows * 128).fill(value),
    },
  };
  yield { kind: "end", port: "audio_embeddings" };
}
async function collect(s: Awaited<ReturnType<typeof setup>>, events = input()) {
  const result = [];
  for await (const x of s.b.run(events, s.signal))
    result.push(...x.tensor.data);
  return result;
}
test("chunked prefill splices audio, brackets cache forwards and decodes greedily", async () => {
  const s = await setup();
  expect(await collect(s)).toEqual([2, 3, 4]);
  expect(s.calls.filter((x) => x.name === "embed").map((x) => x.data)).toEqual([
    [3, 0],
    [7],
    [2],
    [3],
  ]);
  const prefill = s.calls.filter((x) => x.name === "prefill");
  expect(prefill.map((x) => x.shape)).toEqual([
    [1, 2, 128],
    [1, 1, 128],
  ]);
  expect(prefill[0].data![128]).toBe(0.25);
  expect(s.calls.filter((x) => x.name === "forward")).toHaveLength(8);
  await s.b.dispose();
  await s.b.dispose();
  expect(s.cache.dispose).toHaveBeenCalledTimes(1);
});
test("Qwen3 audio placeholder is replaced and its EOS terminates generation", async () => {
  const s = await setup("float16", 3, true);
  expect(await collect(s, input([3, 151675, 7]))).toEqual([2]);
  expect(
    s.calls.filter((call) => call.name === "embed").map((call) => call.data),
  ).toEqual([[3, 0], [7], [2]]);
  const first = s.calls.find((call) => call.name === "prefill");
  expect(first?.data?.[128]).toBe(float32ToHalf(0.25));
  await s.b.dispose();
});
test.each([[151655], [151656], [151652], [-1], [64]])(
  "rejects unsupported token %s",
  async (id) => {
    const s = await setup();
    await expect(collect(s, input([id], 0))).rejects.toThrow(
      "Unsupported Thinker token",
    );
    expect(s.calls.filter((x) => x.name === "embed")).toHaveLength(0);
    await s.b.dispose();
  },
);
test("rejects audio count mismatch before embedding", async () => {
  const s = await setup();
  await expect(collect(s, input([3], 1))).rejects.toThrow("placeholder count");
  await s.b.dispose();
});
test("rejects half overflow before embedding", async () => {
  const s = await setup("float16");
  await expect(collect(s, input([151646], 1, 65520))).rejects.toThrow(
    "overflow float16",
  );
  expect(s.calls.filter((x) => x.name === "embed")).toHaveLength(0);
  await s.b.dispose();
});
test("aborting synchronization prevents output", async () => {
  const s = await setup();
  const c = new AbortController();
  s.device.sync.mockImplementation(async () => {
    c.abort();
  });
  await expect(
    s.b.run(input(), c.signal)[Symbol.asyncIterator]().next(),
  ).rejects.toThrow("aborted");
  await s.b.dispose();
});
test("stop token is excluded and prevents decode", async () => {
  const s = await setup();
  s.config.eos_token_ids = [2];
  expect(await collect(s)).toEqual([]);
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(0);
  await s.b.dispose();
});
test("generation reserves bounded context", async () => {
  const s = await setup("float32", 8);
  expect(await collect(s, input(Array(31).fill(3), 0))).toHaveLength(1);
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(0);
  await s.b.dispose();
});
test("factory remains exclusive until disposal", async () => {
  const s = await setup();
  await expect(
    s.factory.create("thinker", compiled.components.thinker, s.signal),
  ).rejects.toThrow("already in use");
  await s.b.dispose();
  const b = await s.factory.create(
    "thinker",
    compiled.components.thinker,
    s.signal,
  );
  await b.dispose();
});
test("full context rejects prompt before embedding", async () => {
  const s = await setup();
  await expect(collect(s, input(Array(32).fill(3), 0))).rejects.toThrow(
    "prompt tensors",
  );
  expect(s.calls.filter((x) => x.name === "embed")).toHaveLength(0);
  await s.b.dispose();
});
test("abort while paused at output prevents another decode", async () => {
  const s = await setup();
  const c = new AbortController();
  const stream = s.b.run(input(), c.signal)[Symbol.asyncIterator]();
  expect((await stream.next()).done).toBe(false);
  c.abort();
  await expect(stream.next()).rejects.toThrow("aborted");
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(0);
  await s.b.dispose();
});

test.each([1, 3])(
  "cache finalization failure %i releases retained CPU tensors",
  async (endAt) => {
    const s = await setup();
    s.faults.endAt = endAt;
    await expect(collect(s)).rejects.toThrow("cache end failed");
    expect(s.cpuTensors.length).toBeGreaterThan(0);
    for (const tensor of s.cpuTensors)
      expect(tensor.dispose).toHaveBeenCalledTimes(1);
    await s.b.dispose();
  },
);
test.each([1, 2, 5])(
  "scope teardown failure %i releases retained CPU tensors",
  async (failAt) => {
    const s = await setup();
    let scopes = 0;
    s.runtime.endScope = () => {
      if (++scopes === failAt) throw new Error("scope teardown failed");
    };
    await expect(collect(s)).rejects.toThrow("scope teardown failed");
    for (const tensor of s.cpuTensors)
      expect(tensor.dispose).toHaveBeenCalledTimes(1);
    await s.b.dispose();
  },
);
test("cache cleanup preserves an earlier forward failure", async () => {
  const s = await setup();
  s.faults.forward = true;
  s.faults.endAt = 1;
  await expect(collect(s)).rejects.toThrow("original forward failed");
  expect(s.faults.endCalls).toBe(1);
  await s.b.dispose();
});

test("readback cleanup preserves synchronization failure", async () => {
  const s = await setup();
  s.device.sync.mockImplementation(async () => {
    s.cpuTensors.at(-1)!.dispose.mockImplementation(() => {
      throw new Error("dispose failed");
    });
    throw new Error("sync failed");
  });
  await expect(collect(s)).rejects.toThrow("sync failed");
  expect(s.cpuTensors[0].dispose).toHaveBeenCalledTimes(1);
  await s.b.dispose();
});

test.each([false, true])(
  "pending GPU disposal preserves Thinker cache (%s)",
  async (fail) => {
    const s = await setup();
    await verifyPendingTVMDisposal(
      { ...s, component: compiled.components.thinker },
      "thinker",
      input(),
      fail,
    );
  },
);

test.each(["scope", "copy", "microtask"] as const)(
  "queued readback survives %s failure",
  async (kind) => {
    const s = await setup();
    await verifyExceptionalTVMReadback(
      { ...s, component: compiled.components.thinker },
      "thinker",
      input(),
      kind,
    );
  },
);
