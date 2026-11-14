import { readFileSync } from "fs";
import { join } from "path";
import { webcrypto } from "crypto";
import {
  createArtifactGraphExecution,
  GraphComponentFactory,
  GraphInputEvent,
  GraphTensor,
} from "../src/artifact_graph_executor";
import { computeGraphInterfaceId } from "../src/artifact_manifest_v2";

beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  }),
);
const fixture = (name: string) =>
  JSON.parse(
    readFileSync(
      join(__dirname, "fixtures/artifact_manifest_v2", `${name}.json`),
      "utf8",
    ),
  );
const tensor = (
  data: GraphTensor["data"],
  shape = [data.length],
): GraphTensor => ({ data, shape });
async function* stream(...values: GraphTensor[]) {
  yield* values;
}
async function collect(source: AsyncIterable<GraphTensor>): Promise<number[]> {
  const result: number[] = [];
  for await (const t of source) result.push(...t.data);
  return result;
}
function deferred<T = void>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
function inputs() {
  return {
    text: stream(tensor(new Int32Array([1])), tensor(new Int32Array([2]))),
    audio: stream(tensor(new Float32Array([3, 4]))),
    image: stream(tensor(new Uint8Array([5, 0, 0]), [1, 1, 3])),
    video: stream(tensor(new Uint8Array([6, 0, 0]), [1, 1, 1, 3])),
  };
}
function factory() {
  const created: string[] = [],
    disposed: string[] = [],
    events: Record<string, string[]> = {};
  const backend: GraphComponentFactory = {
    create(id, compiled) {
      created.push(id);
      events[id] = [];
      expect(compiled.exports.run).toBe("_run");
      return {
        async *run(source: AsyncIterable<GraphInputEvent>) {
          const values: GraphTensor[] = [];
          let total = 0;
          for await (const event of source) {
            events[id].push(`${event.port}:${event.kind}`);
            if (event.kind !== "data") continue;
            if (id === "thinker" && event.port === "text")
              values.push(event.tensor);
            else total += event.tensor.data[0];
            if (id === "vocoder")
              yield {
                port: "audio",
                tensor: tensor(new Float32Array([event.tensor.data[0]])),
              };
          }
          // EOF flush: encoders emit one value, thinker emits text and hidden state,
          // talker emits codec frames, and vocoder flushes its last waveform block.
          if (id.endsWith("encoder")) {
            const features = new Float32Array(128);
            features[0] = total;
            yield { port: "features", tensor: tensor(features, [1, 128]) };
          } else if (id === "thinker") {
            for (const v of values)
              yield {
                port: "text",
                tensor: tensor(new Int32Array([v.data[0] + total])),
              };
            const hidden = new Float32Array(128);
            hidden[0] = total;
            yield { port: "hidden", tensor: tensor(hidden, [1, 128]) };
          } else if (id === "talker") {
            for (const delta of [0, 1]) {
              const codes = new Int32Array(8);
              codes[0] = total + delta;
              yield { port: "codes", tensor: tensor(codes, [1, 8]) };
            }
          } else if (id === "vocoder")
            yield { port: "audio", tensor: tensor(new Float32Array([-1])) };
        },
        dispose() {
          disposed.push(id);
        },
      };
    },
  };
  return { backend, created, disposed, events };
}
function options(f = factory()) {
  return {
    model: fixture("model"),
    compiled: fixture("compiled"),
    task: "chat.completions",
    inputs: inputs(),
    factory: f.backend,
    queueCapacity: 1,
  };
}

test("mixed inputs execute thinker/talker/codec and flush ordered text/audio", async () => {
  const f = factory();
  const execution = await createArtifactGraphExecution({
    ...options(f),
    outputs: ["text", "audio"],
  });
  const [text, audio] = await Promise.all([
    collect(execution.outputs.text),
    collect(execution.outputs.audio),
  ]);
  await execution.done;
  expect(text).toEqual([15, 16]);
  expect(audio).toEqual([14, 15, -1]);
  expect(f.created.sort()).toEqual([
    "audio_encoder",
    "talker",
    "thinker",
    "vision_encoder",
    "vocoder",
  ]);
  expect(f.disposed.sort()).toEqual(f.created);
  expect(f.events.thinker.filter((e) => e.startsWith("text:"))).toEqual([
    "text:data",
    "text:data",
    "text:end",
  ]);
});

test("omitting optional audio creates and disposes no speech branch", async () => {
  const f = factory(),
    execution = await createArtifactGraphExecution(options(f));
  expect(await collect(execution.outputs.text)).toEqual([15, 16]);
  await execution.done;
  expect(f.created).toEqual(["audio_encoder", "vision_encoder", "thinker"]);
  expect(f.disposed.sort()).toEqual([...f.created].sort());
});

/** Small streaming identity graph isolates queue and cancellation behavior. */
async function identitySetup() {
  const model = fixture("model"),
    compiled = fixture("compiled");
  const p = model.components.thinker.outputs.text;
  model.components = {
    identity: {
      kind: "transform",
      contract: "test.identity.v1",
      inputs: { input: p },
      outputs: { output: p },
      weights: null,
      state: { kind: "none", scope: "request" },
    },
  };
  model.tasks = {
    run: {
      inputs: {
        input: { spec: p, targets: [{ component: "identity", port: "input" }] },
      },
      outputs: {
        output: {
          spec: p,
          source: { component: "identity", port: "output" },
          optional: false,
        },
      },
      edges: [],
    },
  };
  model.interface_id = await computeGraphInterfaceId(model);
  compiled.interface_id = model.interface_id;
  compiled.components = {
    identity: {
      spec: model.components.identity,
      module: "identity.wasm",
      exports: { run: "run" },
      resources: {
        required_features: [],
        max_storage_buffer_binding_size: 0,
        estimated_device_memory_bytes: 0,
      },
    },
  };
  return { model, compiled, task: "run" };
}
function identityFactory(
  onYield = () => {},
  onDispose = () => {},
): GraphComponentFactory {
  return {
    create: () => ({
      async *run(events) {
        for await (const e of events)
          if (e.kind === "data") {
            onYield();
            yield { port: "output", tensor: e.tensor };
          }
      },
      dispose: onDispose,
    }),
  };
}

test("slow consumer applies bounded backpressure and done waits for final drain", async () => {
  const setup = await identitySetup(),
    third = deferred();
  let emitted = 0,
    produced = 0,
    disposed = 0;
  async function* source() {
    for (let i = 0; i < 20; i++) {
      produced++;
      yield tensor(new Int32Array([i]));
    }
  }
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: source() },
    queueCapacity: 1,
    factory: identityFactory(
      () => {
        emitted++;
        if (emitted === 2) third.resolve();
      },
      () => {
        disposed++;
      },
    ),
  });
  await third.promise;
  expect(emitted).toBe(2);
  expect(produced).toBeLessThanOrEqual(5);
  expect(disposed).toBe(0);
  expect(await collect(execution.outputs.output)).toEqual(
    Array.from({ length: 20 }, (_, i) => i),
  );
  await execution.done;
  expect(disposed).toBe(1);
});

test("abort interrupts a pending external next and cleans request state", async () => {
  const setup = await identitySetup(),
    started = deferred(),
    never = new Promise<IteratorResult<GraphTensor>>(() => {});
  let returned = 0,
    requested = 0,
    disposed = 0;
  const source: AsyncIterable<GraphTensor> = {
    [Symbol.asyncIterator]: () => ({
      next: () => {
        if (requested++ === 0)
          return Promise.resolve({
            done: false,
            value: tensor(new Int32Array([9])),
          });
        started.resolve();
        return never;
      },
      return: () => {
        returned++;
        return never;
      },
    }),
  };
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: source },
    factory: identityFactory(undefined, () => {
      disposed++;
    }),
  });
  const iterator = execution.outputs.output[Symbol.asyncIterator]();
  expect((await iterator.next()).value!.data[0]).toBe(9);
  const output = iterator.next();
  await started.promise;
  execution.abort();
  await expect(output).rejects.toThrow(/aborted/);
  await expect(execution.done).rejects.toThrow(/aborted/);
  expect(returned).toBe(1);
  expect(disposed).toBe(1);
});

test("component failure propagates to every output and keeps original disposal error", async () => {
  const setup = await identitySetup();
  let disposed = 0;
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: stream(tensor(new Int32Array([1]))) },
    factory: {
      create: () => ({
        async *run(events) {
          for await (const e of events) {
            if (e.kind === "data") throw new Error("component failed");
            yield { port: "output", tensor: tensor(new Int32Array([0])) };
          }
        },
        dispose() {
          disposed++;
          throw new Error("dispose failed");
        },
      }),
    },
  });
  await expect(collect(execution.outputs.output)).rejects.toThrow(
    "component failed",
  );
  await expect(execution.done).rejects.toThrow("component failed");
  expect(disposed).toBe(1);
});

test("creation failure disposes earlier states", async () => {
  const f = factory(),
    original = f.backend.create;
  f.backend.create = (id, spec, signal) => {
    if (id === "thinker") throw new Error("create failed");
    return original(id, spec, signal);
  };
  await expect(createArtifactGraphExecution(options(f))).rejects.toThrow(
    "create failed",
  );
  expect(f.disposed.sort()).toEqual(["audio_encoder", "vision_encoder"]);
});

test.each([
  tensor(new Float32Array([1])),
  tensor(new Int32Array([1]), [1, 1]),
  tensor(new Int32Array([1]), [2]),
])("invalid tensor rejects and cleans state", async (invalid) => {
  const setup = await identitySetup();
  let disposed = 0;
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: stream(invalid) },
    factory: identityFactory(undefined, () => {
      disposed++;
    }),
  });
  await expect(collect(execution.outputs.output)).rejects.toThrow(/tensor/);
  await expect(execution.done).rejects.toThrow(/tensor/);
  expect(disposed).toBe(1);
});

test("consumer early return aborts the rest of the graph", async () => {
  const setup = await identitySetup();
  let disposed = 0;
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: {
      input: stream(...[1, 2, 3, 4].map((v) => tensor(new Int32Array([v])))),
    },
    queueCapacity: 1,
    factory: identityFactory(undefined, () => {
      disposed++;
    }),
  });
  const iterator = execution.outputs.output[Symbol.asyncIterator]();
  expect((await iterator.next()).done).toBe(false);
  await iterator.return!();
  await expect(execution.done).rejects.toThrow(/aborted/);
  expect(disposed).toBe(1);
});

test("fan-out isolates buffers even when a typed array overrides slice", async () => {
  const setup = await identitySetup();
  const task = setup.model.tasks.run;
  task.outputs.other = JSON.parse(JSON.stringify(task.outputs.output));
  setup.model.interface_id = await computeGraphInterfaceId(setup.model);
  setup.compiled.interface_id = setup.model.interface_id;
  const payload = new Int32Array([7]);
  payload.slice = () => payload;
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: stream(tensor(payload)) },
    factory: identityFactory(),
  });
  const a = execution.outputs.output[Symbol.asyncIterator](),
    b = execution.outputs.other[Symbol.asyncIterator]();
  const [x, y] = await Promise.all([a.next(), b.next()]);
  x.value!.data[0] = 99;
  expect(y.value!.data[0]).toBe(7);
  await Promise.all([a.next(), b.next()]);
  await execution.done;
});

test("abort during pending factory rejects setup and disposes its late result", async () => {
  const setup = await identitySetup(),
    started = deferred(),
    late = deferred<any>(),
    disposed = deferred();
  const controller = new AbortController();
  let disposeCount = 0;
  const creation = createArtifactGraphExecution({
    ...setup,
    inputs: { input: stream() },
    signal: controller.signal,
    factory: {
      create() {
        started.resolve();
        return late.promise;
      },
    },
  });
  await started.promise;
  controller.abort();
  await expect(creation).rejects.toThrow(/abort/i);
  late.resolve({
    run: async function* () {
      yield { port: "output", tensor: tensor(new Int32Array([0])) };
    },
    dispose() {
      disposeCount++;
      disposed.resolve();
    },
  });
  await disposed.promise;
  expect(disposeCount).toBe(1);
});

test("snapshots selected outputs and input stream bindings before asynchronous setup", async () => {
  const f = factory(),
    selected = ["text", "audio"],
    boundInputs = inputs();
  const pending = createArtifactGraphExecution({
    ...options(f),
    inputs: boundInputs,
    outputs: selected,
  });
  selected.pop();
  boundInputs.text = stream(tensor(new Int32Array([1000])));
  const execution = await pending;
  const [text, audio] = await Promise.all([
    collect(execution.outputs.text),
    collect(execution.outputs.audio),
  ]);
  await execution.done;
  expect(text).toEqual([15, 16]);
  expect(audio).toEqual([14, 15, -1]);
});

test("value input cardinality and fixed dimensions are enforced", async () => {
  for (const image of [
    stream(),
    stream(
      tensor(new Uint8Array([1, 2, 3]), [1, 1, 3]),
      tensor(new Uint8Array([1, 2, 3]), [1, 1, 3]),
    ),
    stream(tensor(new Uint8Array([1, 2, 3, 4]), [1, 1, 4])),
  ]) {
    const f = factory(),
      opts = options(f);
    opts.inputs.image = image;
    const execution = await createArtifactGraphExecution(opts);
    await expect(collect(execution.outputs.text)).rejects.toThrow();
    await expect(execution.done).rejects.toThrow();
    expect(f.disposed.sort()).toEqual([...f.created].sort());
  }
});

test("abort after successful drain does not poison EOF", async () => {
  const setup = await identitySetup();
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: stream(tensor(new Int32Array([1]))) },
    factory: identityFactory(),
  });
  const iterator = execution.outputs.output[Symbol.asyncIterator]();
  expect((await iterator.next()).value!.data[0]).toBe(1);
  await execution.done;
  execution.abort();
  expect((await iterator.next()).done).toBe(true);
});

test("premature backend return fails rather than leaving input producers blocked", async () => {
  const setup = await identitySetup();
  let disposed = 0;
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: {
      input: stream(...[1, 2, 3, 4].map((v) => tensor(new Int32Array([v])))),
    },
    queueCapacity: 1,
    factory: {
      create: () => ({
        async *run(events) {
          for await (const e of events)
            if (e.kind === "data") {
              yield { port: "output", tensor: e.tensor };
              return;
            }
        },
        dispose() {
          disposed++;
        },
      }),
    },
  });
  await expect(collect(execution.outputs.output)).rejects.toThrow(
    /before draining inputs/,
  );
  await expect(execution.done).rejects.toThrow(/before draining inputs/);
  expect(disposed).toBe(1);
});

test("spoofed Symbol.toStringTag cannot disguise a tensor dtype", async () => {
  const setup = await identitySetup();
  const payload = new Float32Array([1]);
  Object.defineProperty(payload, Symbol.toStringTag, { value: "Int32Array" });
  const execution = await createArtifactGraphExecution({
    ...setup,
    inputs: { input: stream(tensor(payload)) },
    factory: identityFactory(),
  });
  await expect(collect(execution.outputs.output)).rejects.toThrow(/dtype/);
  await expect(execution.done).rejects.toThrow(/dtype/);
});
