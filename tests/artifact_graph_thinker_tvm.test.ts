import { float32ToHalf } from "../src/artifact_graph_thinker_tvm";
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

import { webcrypto } from "crypto";
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createThinkerTVMFactory } from "../src/artifact_graph_thinker_tvm";
import { loadComponent } from "../src/artifact_graph_tvm";
import { GraphInputEvent } from "../src/artifact_graph_executor";
import { readFileSync } from "fs";
import { join } from "path";
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
  toArray() {
    return this.data.slice();
  }
  view(shape: number[]) {
    const t = new Tensor(shape, this.dtype);
    t.data = this.data;
    return t;
  }
  dispose() {}
}
async function setup(dtype = "float32", maxNewTokens = 3) {
  const calls: { name: string; data?: number[]; shape?: number[] }[] = [];
  const config = {
    model_dtype: dtype,
    hidden_size: 128,
    vocab_size: 64,
    active_vocab_size: 64,
    audio_token_id: 151646,
    image_token_id: 151655,
    video_token_id: 151656,
    eos_token_ids: [151643, 151645],
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
      calls.push({ name: "prefill", shape: x.shape, data: Array.from(x.data) });
      const t = new Tensor([1, 1, 64]);
      t.data[++counter] = 1;
      return { get: () => t };
    },
    decode: (x: Tensor) => {
      calls.push({ name: "decode", shape: x.shape });
      const t = new Tensor([1, 1, 64]);
      t.data[++counter] = 1;
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
  const runtime = {
    beginScope() {},
    endScope() {},
    detachFromCurrentScope: <T>(v: T) => v,
    getParamsFromCacheByName: () => ({ dispose() {} }),
    getGlobalFunc: (name: string) =>
      Object.assign(
        (...args: any[]) => {
          calls.push({ name: name.split("_").at(-1)!, data: args[2] });
        },
        { dispose() {} },
      ),
    makeShapeTuple: (x: number[]) => x,
    scalar: (x: number) => x,
    empty: (shape: number[], dt: string) => new Tensor(shape, dt),
    cpu: () => ({}),
  };
  const device = { sync: jest.fn(async () => {}) };
  const factory = await createThinkerTVMFactory({
    runtime: runtime as unknown as tvmjs.Instance,
    model,
    compiled,
    componentId: "thinker",
    device: device as unknown as tvmjs.DLDevice,
    contextWindowSize: 32,
    maxNewTokens,
    capabilities: { features: new Set(), maxStorageBufferBindingSize: 1e9 },
  });
  const signal = new AbortController().signal;
  const b = await factory.create(
    "thinker",
    compiled.components.thinker as any,
    signal,
  );
  return { b, calls, config, device, cache, factory, signal };
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
  s.b.dispose();
  s.b.dispose();
  expect(s.cache.dispose).toHaveBeenCalledTimes(1);
});
test.each([[151655], [151656], [151652], [-1], [64]])(
  "rejects unsupported token %s",
  async (id) => {
    const s = await setup();
    await expect(collect(s, input([id], 0))).rejects.toThrow(
      "Unsupported Thinker token",
    );
    expect(s.calls.filter((x) => x.name === "embed")).toHaveLength(0);
    s.b.dispose();
  },
);
test("rejects audio count mismatch before embedding", async () => {
  const s = await setup();
  await expect(collect(s, input([3], 1))).rejects.toThrow("placeholder count");
  s.b.dispose();
});
test("rejects half overflow before embedding", async () => {
  const s = await setup("float16");
  await expect(collect(s, input([151646], 1, 65520))).rejects.toThrow(
    "overflow float16",
  );
  expect(s.calls.filter((x) => x.name === "embed")).toHaveLength(0);
  s.b.dispose();
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
  s.b.dispose();
});
test("stop token is excluded and prevents decode", async () => {
  const s = await setup();
  s.config.eos_token_ids = [2];
  expect(await collect(s)).toEqual([]);
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(0);
  s.b.dispose();
});
test("generation reserves bounded context", async () => {
  const s = await setup("float32", 8);
  expect(await collect(s, input(Array(31).fill(3), 0))).toHaveLength(1);
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(0);
  s.b.dispose();
});
test("factory remains exclusive until disposal", async () => {
  const s = await setup();
  await expect(
    s.factory.create("thinker", compiled.components.thinker, s.signal),
  ).rejects.toThrow("already in use");
  s.b.dispose();
  const b = await s.factory.create(
    "thinker",
    compiled.components.thinker,
    s.signal,
  );
  b.dispose();
});
test("full context rejects prompt before embedding", async () => {
  const s = await setup();
  await expect(collect(s, input(Array(32).fill(3), 0))).rejects.toThrow(
    "prompt tensors",
  );
  expect(s.calls.filter((x) => x.name === "embed")).toHaveLength(0);
  s.b.dispose();
});
test("abort while paused at output prevents another decode", async () => {
  const s = await setup();
  const c = new AbortController();
  const stream = s.b.run(input(), c.signal)[Symbol.asyncIterator]();
  expect((await stream.next()).done).toBe(false);
  c.abort();
  await expect(stream.next()).rejects.toThrow("aborted");
  expect(s.calls.filter((x) => x.name === "decode")).toHaveLength(0);
  s.b.dispose();
});
