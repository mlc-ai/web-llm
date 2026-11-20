import { webcrypto } from "crypto";
import { readFileSync } from "fs";
import { join } from "path";
import type * as tvmjs from "@mlc-ai/web-runtime";
import { createThinkerSpeechTVMFactory } from "../src/artifact_graph_thinker_speech_tvm";
import { loadComponent } from "../src/artifact_graph_tvm";
import { GraphInputEvent, GraphTensor } from "../src/artifact_graph_executor";
import {
  verifyExceptionalTVMReadback,
  verifyPendingTVMDisposal,
} from "./tvm_lifecycle_test_utils";

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
const fixture = (name: string) =>
  JSON.parse(
    readFileSync(
      join(__dirname, "fixtures/thinker_speech_tvm", name + ".json"),
      "utf8",
    ),
  );
class Tensor {
  data: Float32Array;
  constructor(
    public shape: number[],
    public dtype = "float32",
  ) {
    this.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
  }
  copyFrom(value: Tensor | Float32Array | Int32Array) {
    this.data.set(value instanceof Tensor ? value.data : value);
    return this;
  }
  toArray() {
    return this.data.slice();
  }
  dispose = jest.fn();
}
async function setup(maxNewTokens = 8) {
  const model = fixture("model"),
    compiled = fixture("compiled"),
    component = compiled.components.thinker;
  const config = {
    model_dtype: "float32",
    hidden_size: 2048,
    vocab_size: 151936,
    active_vocab_size: 151936,
    audio_token_id: 151646,
    image_token_id: 151655,
    video_token_id: 151656,
    eos_token_ids: [151643, 151645],
    context_window_size: 32768,
    prefill_chunk_size: 2,
    speech_profile: {
      speaker_bos_token_ids: [151870, 151872],
      text_eos_token_id: 151861,
      text_pad_token_id: 151859,
    },
  };
  const calls: { name: string; data?: number[] }[] = [];
  let decodes = 0;
  const faults = { forward: false, hiddenShape: false, firstEOS: false };
  const cache = { dispose: jest.fn() };
  const forward = (name: string, input: Tensor) => {
    if (faults.forward) throw Error("original forward failure");
    calls.push({ name, data: Array.from(input.data) });
    const logits = new Tensor([1, 1, 151936]);
    const id = faults.firstEOS
      ? 151645
      : name === "prefill"
        ? 5
        : ++decodes === 1
          ? 6
          : 151645;
    logits.data[id] = 10;
    const hidden = new Tensor(faults.hiddenShape ? [1, 1, 1] : input.shape);
    if (!faults.hiddenShape) hidden.data.set(input.data.map((v) => v + 1));
    return { get: (index: number) => (index === 0 ? logits : hidden) };
  };
  const funcs: Record<string, (...args: any[]) => any> = {
    create_tir_paged_kv_cache: () => cache,
    reset_state: () => calls.push({ name: "reset" }),
    embed: (ids: Tensor) => {
      calls.push({ name: "embed", data: Array.from(ids.data) });
      const result = new Tensor([ids.shape[0], 2048]);
      for (let i = 0; i < ids.data.length; i++)
        result.data.fill(ids.data[i] / 1000, i * 2048, (i + 1) * 2048);
      return result;
    },
    prefill_with_hidden: (x: Tensor) => forward("prefill", x),
    decode_with_hidden: (x: Tensor) => forward("decode", x),
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
  const cpu = {};
  const cpuTensors: Tensor[] = [];
  const runtime = {
    beginScope() {},
    endScope() {},
    detachFromCurrentScope: <T>(v: T) => v,
    getParamsFromCacheByName: () => ({ dispose() {} }),
    getGlobalFunc: (name: string) =>
      Object.assign(() => calls.push({ name }), { dispose: jest.fn() }),
    makeShapeTuple: (x: number[]) => x,
    scalar: (x: number) => x,
    cpu: () => cpu,
    empty: (shape: number[], dtype: string, target: unknown) => {
      const value = new Tensor(shape, dtype);
      if (target === cpu) cpuTensors.push(value);
      return value;
    },
  };
  const device = { sync: jest.fn(async () => {}) };
  const options = {
    runtime: runtime as unknown as tvmjs.Instance,
    device: device as unknown as tvmjs.DLDevice,
    model,
    compiled,
    componentId: "thinker",
    contextWindowSize: 32,
    maxNewTokens,
    speakerBosTokenId: 151870,
    capabilities: {
      features: new Set<string>(),
      maxStorageBufferBindingSize: 2 ** 31,
    },
  };
  const factory = await createThinkerSpeechTVMFactory(options),
    controller = new AbortController();
  const b = await factory.create("thinker", component, controller.signal);
  return {
    b,
    factory,
    controller,
    component,
    config,
    calls,
    faults,
    cache,
    runtime,
    device,
    cpuTensors,
    options,
  };
}
async function* input(
  ids = [3, 151646, 7],
  rows = 1,
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
      shape: [rows, 2048],
      data: new Float32Array(rows * 2048).fill(0.25),
    },
  };
  yield { kind: "end", port: "audio_embeddings" };
}
async function collect(s: Awaited<ReturnType<typeof setup>>, events = input()) {
  const output: Record<string, GraphTensor[]> = {};
  for await (const item of s.b.run(events, s.controller.signal))
    (output[item.port] ??= []).push(item.tensor);
  return output;
}
test("chunked audio prefill emits text, actual aligned states and EOS reason", async () => {
  const s = await setup();
  const out = await collect(s);
  expect(out.token_ids.map((x) => x.data[0])).toEqual([5, 6]);
  expect(out.finish_reason[0].data[0]).toBe(0);
  expect(out.prefix_embeddings[0].shape).toEqual([5, 2048]);
  expect(out.reply_embeddings[0].shape).toEqual([3, 2048]);
  expect(out.prefix_embeddings[0].data[2048]).toBe(1.25);
  expect(out.prefix_embeddings[0].data[3 * 2048]).toBe(Math.fround(151.87));
  expect(out.reply_embeddings[0].data[2048]).toBe(Math.fround(151.861));
  expect(out.reply_embeddings[0].data[2 * 2048]).toBe(Math.fround(151.859));
  expect(s.calls.filter((x) => x.name === "embed").map((x) => x.data)).toEqual([
    [3, 0],
    [7],
    [5],
    [6],
    [151870, 151861, 151859],
  ]);
  await s.b.dispose();
  expect(s.cache.dispose).toHaveBeenCalledTimes(1);
  expect(s.cpuTensors.every((x) => x.dispose.mock.calls.length === 1)).toBe(
    true,
  );
});
test("literal audio token in text remains an ordinary embedding without audio features", async () => {
  const s = await setup();
  const out = await collect(s, input([151646], 0));
  expect(out.prefix_embeddings[0].data[0]).toBe(
    Math.fround(Math.fround(151.646) + Math.fround(Math.fround(151.646) + 1)),
  );
  expect(s.calls.find((x) => x.name === "embed")?.data).toEqual([151646]);
});
test("token cap is explicit and final sampled token is retained in alignment", async () => {
  const s = await setup(2);
  const out = await collect(s, input([3], 0));
  expect(out.token_ids.map((x) => x.data[0])).toEqual([5, 6]);
  expect(out.finish_reason[0].data[0]).toBe(1);
  expect(out.reply_embeddings[0].shape).toEqual([2, 2048]);
});
test("rejects unsupported one-token empty answer rather than fabricating hidden states", async () => {
  const s = await setup();
  s.faults.firstEOS = true;
  await expect(collect(s)).rejects.toThrow("ended before producing text");
  await s.b.dispose();
});
test.each(["float16", "speaker", "badShape", "audioCount"])(
  "rejects %s before returning conditioning",
  async (kind) => {
    const s = await setup();
    if (kind === "float16") {
      s.config.model_dtype = "float16";
      await s.b.dispose();
      await expect(
        s.factory.create("thinker", s.component, s.controller.signal),
      ).rejects.toThrow("FP32");
      return;
    }
    if (kind === "speaker") {
      await expect(
        createThinkerSpeechTVMFactory({
          ...s.options,
          speakerBosTokenId: 42,
        }).then((f) => f.create("thinker", s.component, s.controller.signal)),
      ).rejects.toThrow("metadata");
      return;
    }
    if (kind === "badShape") s.faults.hiddenShape = true;
    await expect(
      collect(s, kind === "audioCount" ? input([3], 1) : input()),
    ).rejects.toThrow();
    await s.b.dispose();
  },
);
test.each([false, true])(
  "pending GPU work drains before disposal (sync failure %s)",
  async (fail) => {
    const s = await setup();
    await verifyPendingTVMDisposal(s, "thinker", input(), fail);
  },
);
test.each(["scope", "copy", "microtask"] as const)(
  "queued readback survives %s failure",
  async (kind) => {
    const s = await setup();
    await verifyExceptionalTVMReadback(s, "thinker", input(), kind);
  },
);
